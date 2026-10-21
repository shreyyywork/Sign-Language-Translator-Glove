# ISL Glove System Audit Report

**Date of Audit:** October 20, 2026  
**Auditor:** Antigravity AI  
**Repository:** `ISL-Glove` (`isl-org/isl-glove-recognition`)  
**Scope:** Full codebase audit covering data ingestion, dataset storage, machine learning pipelines, evaluation protocols, live prediction, speech synthesis, and TinyML C++ export.

---

## 1. Executive Summary

A comprehensive, code-level audit of the **ISL Glove** repository was conducted. While the application possesses a functional FastAPI backend, WebSocket streaming infrastructure, and Scikit-Learn baseline training routines, the audit uncovered **critical defects and methodological flaws** in data provenance, evaluation integrity, feature scaling consistency, and edge deployment:

1. **Silent Fallback to Synthetic Data**: When operating in hardware mode, any missing packet from the serial port silently falls back to generating simulated samples while falsely tagging the transmission as hardware mode.
2. **Synthetic-Only Benchmark Misrepresented as Empirical Evidence**: 100% of existing samples (10,000 rows) are synthetic. The 99.9% accuracy displayed across the UI represents an artificial simulation benchmark, yet is marketed as "Empirically Validated".
3. **Severe Temporal Data Leakage in LSTM**: Overlapping sliding sequence windows (with 83.3% frame overlap) are randomly assigned across train and validation sets, causing cross-contamination.
4. **Data Leakage in Participant-Independent Model Selection**: Model selection tunes against the held-out test participant (U05) without a separate validation participant set.
5. **TinyML C++ Model Divergence**: The exported C++ code evaluates raw sensor values against decision tree thresholds trained on standardized $z$-scores without feature scaling, and discards 99% of Random Forest trees.
6. **Lack of Provenance Tracking**: Datasets do not distinguish whether samples came from `SIMULATED`, `ESP32_USB`, or `ESP32_BLUETOOTH`.

---

## 2. In-Depth Audit Findings (Items 1 to 12)

### Item 1: Where Sensor Data Originates
- **Verified Behavior**:
  - **Simulated Stream**: Originates in `backend/simulator.py` via `GloveSensorSimulator.generate_sample()` (lines 58–172). Uses trigonometric approximations for 20 ISL gestures based on `config/gestures.json`, adding synthetic user biases and Gaussian noise.
  - **Hardware Stream**: Originates in `backend/serial_bluetooth.py` via `SerialBluetoothManager._read_loop()` (lines 146–174) and `_parse_packet()`. It listens on a COM port (USB or Bluetooth SPP) for lines matching `$ISL,...*CRC`.
- **Defects & Risks**:
  - **Critical Fallback Defect (`backend/main.py:414-423`)**:
    ```python
    if system_state["mode"] == "hardware" and serial_manager.is_connected:
        latest = serial_manager.get_latest_sample()
        if latest:
            sample = latest
        else:
            sample = simulator.generate_sample(...) # SILENT SYNTHETIC INJECTION!
    ```
    If the ESP32 lags or drops a frame, the backend substitutes a synthetic sample, while line 448 broadcasts `"mode": "hardware"`. The user and the recording buffer have no way of knowing synthetic data was injected into a hardware session.
- **Recommended Fix**: Remove the silent simulator fallback. When in hardware mode, if no valid packet arrives within timeout, flag the frame as `NO_HARDWARE_PACKET` or wait for the hardware buffer. Never fabricate sensor readings.

---

### Item 2: Whether Simulated Data is Distinguishable from Hardware Data
- **Verified Behavior**:
  - In individual Python dictionaries, `simulator.py` adds `"is_simulated": True` and `serial_bluetooth.py` sets `"is_simulated": False`.
  - In CSV files, a boolean column `is_simulated` exists.
- **Defects & Risks**:
  - A boolean flag is insufficient: it cannot distinguish between `ESP32_USB` and `ESP32_BLUETOOTH`.
  - In `ml/dataset_manager.py` (lines 43–76), `load_dataset("all")` merges files from `data/simulated/` and `data/real/` without source isolation.
  - Hardware data consists of 12-bit ADC integers ($1000$ to $3500$), while simulated data consists of normalized floats ($0.0$ to $1.0$). If mixed together, `preprocessor.py` (line 56) gets corrupted by disparate ranges.
- **Recommended Fix**: Introduce an explicit provenance enum: `SIMULATED`, `ESP32_USB`, and `ESP32_BLUETOOTH`. Prohibit pooling synthetic and real data into training batches without explicit provenance separation.

---

### Item 3: How CSV Samples are Generated and Stored
- **Verified Behavior**:
  - **Batch Generation**: `GloveSensorSimulator.generate_full_dataset()` writes to `data/simulated/isl_simulated_dataset.csv`.
  - **Live Recording**: `frontend/js/app.js` captures frames during a 3s/5s/10s window and sends them to `POST /api/dataset/record`.
  - **CSV Append**: `DatasetManager.append_recording_session()` in `ml/dataset_manager.py` appends samples into `data/real/isl_real_dataset.csv` or `data/simulated/isl_simulated_dataset.csv`.
- **Defects & Risks**:
  - Raw sensor features are overwritten with uncalibrated or normalized values depending on mode.
  - No recording metadata is recorded: session IDs, timestamps of session start, and packet sequence numbers are omitted.
- **Recommended Fix**: Add `session_id`, `source`, `raw_adc` features, and separate calibrated values from raw readings in the CSV schema.

---

### Item 4: Whether Generated Samples are Synthetic or Measured
- **Verified Behavior**:
  - Inspection of `data/real/` reveals that it is **completely empty** (0 files, 0 samples).
  - Inspection of `data/simulated/isl_simulated_dataset.csv` reveals **10,000 rows, all 100% synthetic**.
- **Defects & Risks**:
  - All existing models, evaluations, and confusion matrices were trained and validated **strictly on synthetic mathematical curves**, not measured human kinematics.
- **Recommended Fix**: Label all synthetic-derived models and metrics as "Simulation Benchmark". Display zero real samples in UI until physical hardware is actually recorded.

---

### Item 5: How Train / Validation / Test Splits are Performed
- **Verified Behavior**:
  - **Stratified Split**: `sklearn.model_selection.train_test_split(df, test_size=0.2, stratify=df["gesture"])` in `dataset_manager.py:164`.
  - **User-Independent Split**: `train_df = df[df["user_id"].isin(train_users)]`, `test_df = df[df["user_id"].isin(test_users)]` in `dataset_manager.py:146`.
  - **LSTM Split**: Overlapping sliding sequences randomly split via `torch.utils.data.random_split(dataset, [train_size, val_size])` in `dynamic_lstm.py:104`.
- **Defects & Risks**:
  - In LSTM training, adjacent overlapping windows share 25 frames, corrupting validation accuracy.
  - In user-independent split, only two sets exist: Train (U01–U04) and Test (U05). No Validation participant set exists.
- **Recommended Fix**: Implement a 3-way participant split: Train Participants, Validation Participants (for model tuning/selection), and strictly held-out Test Participants (for final reporting). For time-series, split continuous sessions without window overlap across folds.

---

### Item 6: Whether User-Independent Evaluation is Genuinely Implemented
- **Verified Behavior**:
  - The code physically filters `train_users` and `test_users` by `user_id`.
- **Defects & Risks**:
  - **Model Selection Peeking**: Line 154 of `ml/train_baseline.py`:
    ```python
    best_model_name = max(results.keys(), key=lambda k: results[k]["f1_weighted"])
    ```
    `results[k]` was computed on `test_df` (U05). The test set was used to select the winning model! This violates participant independence.
  - Furthermore, `U05` is an artificial profile created by `simulator.py` with an arbitrary fixed offset, not a distinct human subject.
- **Recommended Fix**: Split participants into Train (e.g. U01, U02), Validation (e.g. U03, U04), and Final Test (e.g. U05). Perform model selection solely on Validation participants.

---

### Item 7: Whether Data Leakage is Possible
- **Verified Behavior**:
  - **Temporal Leakage**: Yes, confirmed in `ml/dynamic_lstm.py` (lines 86–104). Overlapping windows are randomly shuffled into train and test sets.
  - **Preprocessor Leakage**: In `ml/preprocessor.py:56`, `clean_dataframe()` computes `df.max()` on the merged dataframe prior to splitting.
  - **Hyperparameter Leakage**: The held-out test participant is accessed to rank and pick the best baseline model.
- **Recommended Fix**: Fit scalers and feature transformers strictly on training participants. Perform session-level chronological sequence windowing for time series.

---

### Item 8: Whether Displayed Accuracy Comes from Actual Model Evaluation
- **Verified Behavior**:
  - The metrics stored in `models/evaluation/model_comparison.json` ($99.9\%$ accuracy, $1.00$ F1) were indeed computed by Scikit-Learn evaluation functions.
- **Defects & Risks**:
  - Because the dataset was generated using rigid trigonometric equations with narrow Gaussian noise, the 20 classes are artificially distinct in 11D space. Achieving 99.9% on synthetic data is trivial and does not reflect real-world sensor noise or anatomical variability.
  - Displaying this as "Empirically Validated" misleads users.
- **Recommended Fix**: Change UI labels to "Synthetic Simulation Benchmark". If insufficient real participants exist, display `"Insufficient real data for participant-independent evaluation"`.

---

### Item 9: Whether Live Prediction Uses the Trained Model or Simulator Rules
- **Verified Behavior**:
  - `backend/prediction_service.py` (lines 84–112) passes incoming sensor features to `self.model_selector.active_model.predict_proba()` or `predict()`.
  - It does **not** rely on simulator-specific heuristic rules.
- **Defects & Risks**:
  - If dynamic LSTM is trained, `prediction_service.py` cannot run it: the service has no sequence buffer and only interfaces with static Scikit-Learn `.joblib` baselines.
- **Recommended Fix**: Preserve model-driven inference. Add a circular sliding-window feature buffer in `prediction_service.py` to enable dynamic LSTM sequence inference when selected.

---

### Item 10: Whether Confidence Values are Genuine Model Outputs
- **Verified Behavior**:
  - For models implementing `predict_proba()` (SVM, Random Forest, Decision Tree), confidence is the actual posterior probability: `float(probs[top_idx])`.
- **Defects & Risks**:
  - Fallback in `prediction_service.py:111`: If a model lacks `predict_proba`, it returns a hardcoded dummy confidence `0.85`.
- **Recommended Fix**: When `predict_proba` is unavailable, use calibrated decision function distance or flag confidence as `UNAVAILABLE`, rather than fabricating an artificial 85%.

---

### Item 11: Whether the LSTM Trains on Actual Sequences
- **Verified Behavior**:
  - `ml/preprocessor.py:create_sliding_sequences()` extracts 3D tensors of shape `(N, 30, 11)`.
  - `DynamicLSTMClassifier` in `ml/dynamic_lstm.py` runs a 2-layer LSTM on these sequences.
- **Defects & Risks**:
  - **Unscaled Features**: In `DynamicLSTMPipeline.train_lstm()` (lines 85–90), `preprocessor = SensorPreprocessor()` is instantiated, but `preprocessor.fit()` is never called before sequence generation. `X_seq` is trained on unstandardized raw numbers!
  - **Validation Contamination**: Overlapping windows cross between train and val splits.
  - **Disconnected from Runtime**: The trained weights (`lstm_model.pth`) are never loaded by the live WebSocket server.
- **Recommended Fix**: Fit scalers on train sequence participants. Split whole gesture continuous takes, not overlapping frames. Connect sequence inference into the live prediction service.

---

### Item 12: Whether TinyML Export Produces Equivalent Predictions
- **Verified Behavior**:
  - `ml/tinyml_exporter.py` generates C++ source and header files (`TinyML_Classifier.h` and `.cpp`).
- **Critical Defects & Risks**:
  1. **Scaling Incompatibility**: The Scikit-Learn Decision Tree is trained on standardized $z$-scores ($X_{\text{scaled}} = (X - \mu)/\sigma$). The exported C++ compares unstandardized features `features[i] <= threshold` directly against standardized split thresholds! Every single decision branch evaluates wrong numerical scales.
  2. **Random Forest Truncation (`tinyml_exporter.py:88`)**:
     ```python
     if isinstance(clf, RandomForestClassifier):
         tree = clf.estimators_[0].tree_
     ```
     It discards 99 of the 100 trees in the ensemble and only exports the first tree, completely breaking ensemble accuracy.
- **Recommended Fix**:
  - Incorporate scaler mean ($\mu$) and standard deviation ($\sigma$) arrays directly into the generated C++ header so inputs are scaled on-chip before tree evaluation.
  - Or, train a dedicated Decision Tree specifically on raw/min-max features without standardization for TinyML export.
  - Prohibit single-tree export for Random Forest, or generate a true lightweight multi-tree voting array in C++.

---

## 3. Comprehensive Defect & Risk Matrix

| # | Component | Defect Description | Severity | Risk |
|---|---|---|---|---|
| **D1** | `backend/main.py` | Silent fallback to simulator when hardware packet missing | **CRITICAL** | Synthetic data silently corrupts real hardware sessions |
| **D2** | `ml/dataset_manager.py` | Missing provenance enum (`SIMULATED`, `ESP32_USB`, `ESP32_BLUETOOTH`) | **HIGH** | Inability to track data origin; synthetic/real data pooling |
| **D3** | `ml/dynamic_lstm.py` | Random split of overlapping sliding windows | **CRITICAL** | Severe temporal data leakage; invalid validation metrics |
| **D4** | `ml/dynamic_lstm.py` | Features trained without scaler standardization | **HIGH** | Degraded numerical stability and convergence |
| **D5** | `ml/train_baseline.py` | Held-out test participant used for model selection | **HIGH** | Overfitting to test participant; biased generalizability claims |
| **D6** | `ml/tinyml_exporter.py` | Exported C++ lacks feature scaling; compares raw values to $z$-score thresholds | **CRITICAL** | Zero prediction equivalence on ESP32 hardware |
| **D7** | `ml/tinyml_exporter.py` | Random Forest exports only 1 tree out of 100 | **HIGH** | Completely breaks ensemble logic |
| **D8** | `backend/serial_bluetooth.py` | Packet parser discards CRC checksum without verification | **MEDIUM** | Corrupted packets can enter the prediction pipeline |
| **D9** | `frontend/` & Reports | 10,000 synthetic samples marketed as "Empirical 99.9%" | **HIGH** | Misleading performance claims; non-compliance with scientific reporting |
| **D10** | `config/gestures.json` | Missing official references, one/two-hand tags, and hardware support flags | **MEDIUM** | Inability to verify authentic ISL signs against ISLRTC standards |

---

## 4. Remediation Plan Across Remaining Phases

1. **Phase 2 (Data Provenance)**:
   - Introduce `DataSource` enum (`SIMULATED`, `ESP32_USB`, `ESP32_BLUETOOTH`).
   - Store explicit provenance in every sample and CSV.
   - Separate real and synthetic datasets; update UI with genuine counts and "Simulation Benchmark" labels.
2. **Phase 3 (Real Sensor Ingestion)**:
   - Robust USB serial reader with XOR checksum verification, packet drop counters, timeout handling, and missing-packet alarms.
3. **Phase 4 (Data Collection)**:
   - Schema upgrade: `user_id`, `session_id`, `gesture`, `timestamp`, `source`, 11 raw sensor features, and separated normalized values.
4. **Phase 5 (Model Evaluation)**:
   - Proper participant-independent splitting: Train Users (e.g. U01, U02), Validation Users (U03, U04), Test Users (U05).
   - Display `"Insufficient real data for participant-independent evaluation"` when real participant data is absent.
5. **Phase 6 (Live Prediction)**:
   - Temporal smoothing, strict confidence thresholds, explicit `UNKNOWN` handling, debouncing, and duplicate speech suppression.
6. **Phase 7 (ISL Vocabulary)**:
   - Enhance `config/gestures.json` with ISLRTC official references, static/dynamic classification, one/two-hand requirements, and hardware suitability.
7. **Phase 8 (Tests)**:
   - Automated tests for provenance, checksum validation, leakage checks, preprocessor scaling, TinyML equivalence, and temporal smoothing.
