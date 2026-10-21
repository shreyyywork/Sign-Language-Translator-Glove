# ISL Glove Web Dashboard – Comprehensive Technical Documentation

An in-depth guide to the **Indian Sign Language (ISL) Recognition Web Application**, documenting the system architecture, frontend and backend technology stacks, communication protocols, and an exhaustive inventory of every user interface component and its functional purpose.

---

## 📋 Table of Contents
1. [Executive Overview & Purpose](#1-executive-overview--purpose)
2. [High-Level Architecture & Data Flow](#2-high-level-architecture--data-flow)
3. [Technology Stack](#3-technology-stack)
   - [Frontend Technologies](#frontend-technologies)
   - [Backend & Serving Technologies](#backend--serving-technologies)
   - [Machine Learning & Signal Processing](#machine-learning--signal-processing)
   - [Firmware & Hardware Protocols](#firmware--hardware-protocols)
4. [File & Directory Structure](#4-file--directory-structure)
5. [Complete Inventory of UI Components & Uses](#5-complete-inventory-of-ui-components--uses)
   - [5.1 Global Top Navigation Header](#51-global-top-navigation-header)
   - [5.2 GitHub Underline Subnavigation Bar](#52-github-underline-subnavigation-bar)
   - [5.3 Tab 1: Dashboard](#53-tab-1-dashboard)
   - [5.4 Tab 2: Live Translation Kiosk](#54-tab-2-live-translation-kiosk)
   - [5.5 Tab 3: Dataset Collection System](#55-tab-3-dataset-collection-system)
   - [5.6 Tab 4: 3-Step Calibration Wizard](#56-tab-4-3-step-calibration-wizard)
   - [5.7 Tab 5: Model Training Engine](#57-tab-5-model-training-engine)
   - [5.8 Tab 6: Model Evaluation & Metrics](#58-tab-6-model-evaluation--metrics)
   - [5.9 Tab 7: ISL Vocabulary Explorer](#59-tab-7-isl-vocabulary-explorer)
   - [5.10 Tab 8: Real-Time Translation History](#510-tab-8-real-time-translation-history)
   - [5.11 Tab 9: Hardware Settings & TinyML Exporter](#511-tab-9-hardware-settings--tinyml-exporter)
   - [5.12 Global Notification System (Toasts)](#512-global-notification-system-toasts)
6. [Real-Time WebSocket Protocol & Telemetry Schema](#6-real-time-websocket-protocol--telemetry-schema)
7. [Speech Synthesis & Sentence Construction Engine](#7-speech-synthesis--sentence-construction-engine)
8. [Setup, Execution & Verification](#8-setup-execution--verification)

---

## 1. Executive Overview & Purpose

The **ISL Glove Web Dashboard** is a real-time web application serving as the human-machine interface (HMI) for a wearable IoT assistive glove. The glove translates non-verbal **Indian Sign Language (ISL)** gestures into written text and spoken voice in both **English** and **Hindi**.

### Core Objectives of the Web Application:
- **Real-Time Telemetry Streaming**: Visualizes 5-finger flexion and 6-DOF spatial acceleration/angular velocity at **50 Hz** (20ms latency).
- **Interactive 3D Hand Skeletal Visualizer**: Renders hand postures dynamically on an HTML5 Canvas using forward kinematics and sensor readings.
- **Dual Operating Modes**: Allows seamless switching between a physical ESP32 hardware glove (Bluetooth/USB) and a built-in **Biomechanical Hand Sensor Simulator** for software-only development and testing.
- **On-The-Fly ML Inference**: Displays gesture predictions, prediction probabilities, debounced confidence scoring, and top-3 candidate classifications.
- **Sentence Builder & Multilingual TTS**: Automatically concatenates recognized signs into conversational sentences and speaks them aloud using the browser's Web Speech API in English or Hindi.
- **Per-User Calibration Wizard**: Provides an interactive 3-step wizard to eliminate flex sensor physical drift and calibrate per-user finger length differences.
- **End-to-End Dataset Management**: Facilitates recording multi-user gesture datasets, generating simulated data, and inspecting sample class distributions.
- **No-Code Model Training & Evaluation**: Enables training Decision Tree, Random Forest, SVM, and PyTorch LSTM models directly from the browser, with live confusion matrices and evaluation reports.
- **Edge Deployment (TinyML)**: Generates zero-dependency C++ code directly from trained Scikit-Learn models for execution directly on the ESP32 microcontroller.

---

## 2. High-Level Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                                PHYSICAL SENSOR GLOVE                              |
|   5 Flex Sensors (GPIO 32, 33, 34, 35, 39) + MPU-6050 6-DOF IMU (I2C: GPIO 21, 22)  |
|                                         |                                         |
|                 ESP32 Microcontroller (Bluetooth SPP / USB COM)                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          | Raw 11-Feature Packet (50Hz)
                                          v
+-----------------------------------------------------------------------------------+
|                         BACKEND ENGINE (FastAPI / Uvicorn)                        |
|  - Hardware Serial Reader / Biomechanical Sensor Simulator                        |
|  - 3-Step Per-User Normalizer (Relaxed / Straight / Bent)                         |
|  - Standard Scaler & Preprocessor Pipeline                                        |
|  - Real-Time Inference (Decision Tree / Random Forest / SVM / PyTorch LSTM)        |
|  - Confidence Debouncer & Multilingual Translation Mapper                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          | WebSocket JSON Stream: ws://127.0.0.1:8000/ws/stream
                                          v
+-----------------------------------------------------------------------------------+
|                                  WEB FRONTEND                                     |
|  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────┐  |
|  │  2D Skeletal Visualizer │ │ Live Hero Display       │ │ Sentence Builder    │  |
|  │  HTML5 Canvas           │ │ Hindi / English Display │ │ Web Speech TTS      │  |
|  └─────────────────────────┘ └─────────────────────────┘ └─────────────────────┘  |
|  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────┐  |
|  │  Dataset Recording      │ │ 3-Step Calibration      │ │ Model Training & CM │  |
|  │  Multi-User CSV Export  │ │ Normalization Wizard    │ │ GitHub Matrix Graph │  |
|  └─────────────────────────┘ └─────────────────────────┘ └─────────────────────┘  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

### Frontend Technologies

| Technology | Purpose | Implementation Details |
|---|---|---|
| **HTML5 (Semantic)** | Page Structure & Layout | Single Page Application (SPA) structure, accessibility attributes, data tags, and modal containers. |
| **Vanilla CSS3** | Visual Design & Styling | Complete custom design system based on **GitHub Primer Dark** palette (`#0d1117` canvas, `#161b22` surfaces, `#30363d` borders, `#238636` success green). Implements responsive CSS Grid, Flexbox, custom scrollbars, and keyframe pulse animations. |
| **Modern ECMAScript (ES6+)** | Application Logic & State | Object-oriented modular architecture (`ISLGloveApp`, `HandVisualizer`, `MetricsVisualizer`). Uses native `fetch`, `async`/`await`, and WebSocket event handlers without heavyweight frontend framework overhead. |
| **HTML5 Canvas 2D API** | Real-Time Hand Visualizer | Hardware-accelerated 2D canvas running at 60 FPS via `requestAnimationFrame`. Renders palm plate, joint pivots, and 5 multi-joint fingers with dynamic color flexion mapping. |
| **WebSockets API** | High-Frequency Real-Time Telemetry | Full-duplex WebSocket client (`ws://`) receiving 50 Hz JSON sensor packets and predictions with auto-reconnection logic. |
| **Web Speech API (`SpeechSynthesis`)** | Text-to-Speech (TTS) | Client-side native speech synthesis using `window.speechSynthesis` and `SpeechSynthesisUtterance`. Dynamically discovers English and Hindi system voices. |

### Backend & Serving Technologies

| Technology | Purpose | Implementation Details |
|---|---|---|
| **Python 3.11 – 3.14+** | Backend Runtime Environment | Core server execution, mathematical computations, and sensor management. |
| **FastAPI (0.142+)** | REST API & WebSocket Server | High-performance asynchronous web framework providing Swagger documentation (`/docs`), OpenAPI schemas, and WebSocket endpoint `/ws/stream`. |
| **Uvicorn (0.42+)** | ASGI Web Server | Lightning-fast asynchronous server process running on `127.0.0.1:8000`. |
| **Starlette StaticFiles** | Asset Serving | Mounts `/static` for serving HTML, CSS, JavaScript, and icons. |
| **Pydantic (v2)** | Data Validation & Schemas | Strongly typed request and response validation for API payloads. |

### Machine Learning & Signal Processing

| Library | Purpose | Implementation Details |
|---|---|---|
| **Scikit-Learn (1.9+)** | Baseline Classification | Decision Tree, Random Forest (100 estimators), and Support Vector Machine (SVM with RBF non-linear kernel). |
| **PyTorch (2.14+)** | Dynamic Sequence Recognition | Multi-layer bidirectional LSTM neural network evaluating 30-step sliding time-series windows for dynamic gestures (waving, shaking, nodding). |
| **NumPy & Pandas** | Signal Processing & Dataframes | Vectorized sensor normalization, statistical outlier filtering, and dataset management. |
| **Joblib** | Model Persistence | Serializes and deserializes trained models, standard scalers, and label encoders to `models/saved/`. |
| **Custom TinyML Exporter** | Edge Firmware Generation | Translates trained Decision Tree rule paths into zero-dependency C++ header files (`ISL_TinyML_Model.h`) for on-ESP32 inference (<1ms). |

### Firmware & Hardware Protocols

| Component | Specification |
|---|---|
| **Microcontroller** | ESP32 DevKit v1 Dual-Core Tensilica Xtensa 32-bit LX6 @ 240MHz. |
| **Sensors** | 5 Flex Resistance Divider Sensors (GPIO 32, 33, 34, 35, 39) + MPU-6050 6-DOF IMU (I2C: GPIO 21 SDA, GPIO 22 SCL). |
| **Serial Protocol** | 115,200 Baud Rate, 50 Hz packet rate, comma-separated 11-feature frame format with XOR checksum. |
| **Bluetooth Protocol** | Bluetooth Classic 2.0 SPP (Serial Port Profile), broadcasting as `ISL-Glove-ESP32`. |

---

## 4. File & Directory Structure

The website and associated client assets are structured as follows:

```
frontend/
├── index.html              # Main Single Page Application (GitHub Primer Dark theme)
├── css/
│   └── style.css           # 847-line custom CSS design system & component token definitions
└── js/
    ├── app.js              # 764-line main controller: WebSocket, state, tabs, TTS, REST API client
    ├── visualizer.js       # 205-line HTML5 Canvas 2D kinematic hand renderer (60 FPS)
    └── charts.js           # 129-line metric visualizer: GitHub contribution graph confusion matrix & tables
```

---

## 5. Complete Inventory of UI Components & Uses

Every component across the entire web application is detailed below by its visual location, DOM identifier, and functional role.

---

### 5.1 Global Top Navigation Header

The top bar persists across all pages, modeled after GitHub’s navigation bar.

| Component Name | Element ID / Class | Visual Description | Purpose & Functional Use |
|---|---|---|---|
| **Logo & Badge** | `.gh-logo-badge` | Glove emoji `🧤` on GitHub dark badge | Provides visual identity and branding for the project. |
| **Repository Breadcrumb** | `.gh-repo-breadcrumb` | `isl-org / isl-glove-recognition [Public]` | Displays project workspace namespace with GitHub public repository styling. |
| **Mode Segmented Toggle** | `#btnSimMode`, `#btnHwMode` | Segmented button group with glowing status dot | Toggles between **Simulator Mode (50Hz)** and **ESP32 Hardware Mode**. Sends POST request to `/api/mode` to switch backend telemetry source. |
| **Status Mode Dot** | `#statusModeDot` | Animated circular pulse indicator | Green pulse for active simulation; changes to blue/amber when connected to hardware. |
| **TTS Audio Button** | `#btnToggleTTS` | Green button with speaker icon (`🔊 Audio TTS: ON`) | Toggles automatic browser Text-to-Speech voice synthesis on or off for recognized signs. |
| **WebSocket Badge** | `#wsStatusBadge`, `#wsStatusDot` | GitHub Primer Success/Danger label | Displays live connection status to `ws://127.0.0.1:8000/ws/stream` (`Stream Live`, `Connecting...`, or `Offline`). |
| **Swagger API Quick Link** | `.gh-segment-btn` | Button with book icon `📖 Swagger API` | Opens the FastAPI interactive API documentation (`/docs`) in a new browser tab. |
| **Export TinyML Button** | `.gh-segment-btn` | Button with lightning icon `⚡ Export TinyML C++` | Triggers immediate conversion of the active Scikit-Learn model into an ESP32 C++ header file. |

---

### 5.2 GitHub Underline Subnavigation Bar

A GitHub-style navigation bar (`.gh-underline-nav`) providing access to all 9 application modules without page reloads.

| Tab Item | Data Attribute | Icon & Label | Counter Badge | Purpose |
|---|---|---|---|---|
| **Dashboard** | `data-tab="dashboard"` | 📊 Dashboard | — | Primary monitoring center: 3D hand visualizer, gauges, IMU values, hero prediction card, and simulator triggers. |
| **Live Translation** | `data-tab="live"` | ⚡ Live Translation | — | Kiosk-mode display for non-verbal users with oversized typography and speech settings. |
| **Dataset Collection** | `data-tab="collection"` | 📁 Dataset Collection | `#navDatasetCount` (10k) | Multi-user CSV recorder, duration timer, and baseline dataset generator. |
| **Calibration** | `data-tab="calibration"` | 🎯 Calibration | — | 3-step wizard to calibrate neutral, straight, and bent sensor limits per user. |
| **Model Training** | `data-tab="training"` | 🧠 Model Training | — | Train Decision Tree, Random Forest, SVM, and PyTorch LSTM models. |
| **Model Evaluation** | `data-tab="evaluation"` | 📈 Model Evaluation | `#navEvalF1` (100%) | Empirical performance metrics, normalized confusion matrix, and per-class reports. |
| **ISL Vocabulary** | `data-tab="vocabulary"` | 📖 ISL Vocabulary | `20` | Interactive dictionary of the 20 predefined Indian Sign Language signs. |
| **Translation History** | `data-tab="history"` | 📜 Translation History | — | Chronological log of recognized signs with timestamps, confidence scores, and audio replay. |
| **Settings** | `data-tab="settings"` | ⚙️ Settings | — | ESP32 COM port discovery, Bluetooth pairing guide, and TinyML export console. |

---

### 5.3 Tab 1: Dashboard

The central real-time control room of the system.

#### A. Constructed Phrase Banner (`.gh-sentence-bar`)
- **Phrase Display (`#currentSentenceText`)**: Displays concatenated signs formed during natural conversation (e.g., `"Hello Welcome Please"`).
- **Speak Phrase Button (`#btnSpeakSentence`)**: Synthesizes speech for the entire constructed sentence in the selected language.
- **Clear Button (`#btnClearSentence`)**: Flushes the phrase buffer and clears the text display.

#### B. Hand Skeletal Kinematics Card
- **Canvas Visualizer (`#handCanvas`)**: Renders palm plate, wrist joint, and 5 articulated fingers. Hand pitch and roll adjust in real time from IMU accelerometer gravity vectors.
- **Visualizer Legend (`.visualizer-legend`)**: Color-codes finger deflection:
  - 🔵 **Straight** (<25% flex)
  - 🟢 **Mid** (25%–55% flex)
  - 🟡 **High** (55%–80% flex)
  - 🔴 **Fist / Closed** (>80% flex)
- **5 Vertical Flex Gauges**:
  - `Thumb` (`#flex-thumb-fill`, `#flex-thumb-val`)
  - `Index` (`#flex-index-fill`, `#flex-index-val`)
  - `Middle` (`#flex-middle-fill`, `#flex-middle-val`)
  - `Ring` (`#flex-ring-fill`, `#flex-ring-val`)
  - `Little` (`#flex-little-fill`, `#flex-little-val`)
  - Displays real-time vertical fill bars (0% to 100%) representing finger flexion.
- **MPU6050 Coordinate Boxes**:
  - **Accelerometer**: Shows instantaneous values for `acc_x`, `acc_y`, `acc_z` in gravitational units ($g$).
  - **Gyroscope**: Shows angular rotational velocities for `gyro_x`, `gyro_y`, `gyro_z` in radians per second ($\text{rad/s}$).

#### C. Recognition Hero Card (`.gh-prediction-hero`)
- **English Gesture Name (`#dashPredictedGesture`)**: Large font display of recognized sign (e.g., `HELLO`, `THANK_YOU`, `HELP`).
- **Hindi Translation (`#dashHindiGesture`)**: Devanagari script translation (e.g., `नमस्ते`, `धन्यवाद`, `मदद`).
- **Confidence Progress Bar (`#dashConfidenceFill`, `#dashConfidencePct`)**: Green progress bar indicating the classification probability (e.g., `95.0%`).
- **Active Model Badge (`#dashModelBadge`)**: Identifies currently active inference algorithm (e.g., `Model: SVM (Empirically Validated)`).
- **Top-3 Candidate Pills (`#dashTopKPills`)**: Displays the top 3 gesture predictions ranked by confidence probability.

#### D. Biomechanical Sensor Simulator Controller
- **Gesture Selector (`#simGestureSelect`)**: Dropdown menu containing all 20 predefined ISL gestures. Changing this simulates realistic anatomical angles, flexion noise, and IMU gravity vectors for that sign at 50 Hz.
- **User Anatomical Profile Selector (`#simUserSelect`)**:
  - `U01`: Baseline user profile
  - `U02`: High flexibility user
  - `U03`: Tremor bias (motor impairment testing)
  - `U04`: Tight grip user
  - `U05`: Held-out test subject (used for user-independent validation)

---

### 5.4 Tab 2: Live Translation Kiosk

Designed for fullscreen kiosk deployment or public communication between non-verbal individuals and hearing conversation partners.

| Component Name | Element ID | Purpose |
|---|---|---|
| **Oversized Gesture Banner** | `#livePredictedGesture` | 56px high-contrast typography displaying the recognized sign. |
| **Oversized Hindi Banner** | `#liveHindiGesture` | 32px Devanagari translation for regional bilingual communication. |
| **Confidence Badge** | `#liveConfidencePct` | Large green pill badge indicating classification confidence. |
| **Speech Language Selector** | `#ttsLangSelect` | Dropdown to switch voice speech between English (`en`) and Hindi (`hi`). |
| **Conversation Buffer Controller** | Inline actions | Buttons to trigger voice synthesis of the constructed phrase or clear the conversation history. |

---

### 5.5 Tab 3: Dataset Collection System

Provides a complete workflow for gathering, validating, and generating training datasets.

#### A. Summary Metric Cards
- **Total Samples (`#statTotalSamples`)**: Cumulative rows across all datasets.
- **Real Hardware Samples (`#statRealSamples`)**: Count of samples captured from physical ESP32 glove.
- **Simulated Samples (`#statSimSamples`)**: Count of biomechanically generated samples.
- **Distinct User Subjects (`#statUserCount`)**: Total number of unique subject IDs (`U01`, `U02`, etc.) in the dataset.

#### B. Session Recording Panel
- **Subject Selector (`#recordUserSelect`)**: Selects target subject ID (`U01` to `U05`).
- **Target Gesture Selector (`#recordGestureSelect`)**: Selects the sign being recorded.
- **Session Duration (`#recordDurationSelect`)**: Options for `3 Seconds` (~150 samples), `5 Seconds` (~250 samples), or `10 Seconds` (~500 samples).
- **Start Recording (`#btnStartRecord`)**: Initiates 50 Hz streaming capture into memory buffer with live countdown display (`#recordingStatusBadge`, `#recordCountdown`).
- **Stop Recording (`#btnStopRecord`)**: Aborts or finalizes recording early, automatically persisting valid rows into standard CSV files.
- **Sample Counter (`#recordSampleCounter`)**: Live counter updating with every received packet during recording.
- **Generate Baseline Dataset Button (`#btnGenerateSimDataset`)**: One-click synthetic data generation of **10,000 multi-user samples** across all 20 ISL gestures with anatomical Gaussian noise.

#### C. Class Distribution Breakdown Table (`#datasetDistributionBody`)
- Lists each gesture class, the total number of samples collected, and an inline visual distribution bar representing dataset balance.

---

### 5.6 Tab 4: 3-Step Calibration Wizard

Eliminates flex sensor physical drift and anatomical variance by recording 3 baseline hand postures.

```
+-----------------------------------------------------------------------------------+
|                           3-STEP CALIBRATION PROTOCOL                             |
|                                                                                   |
|  [ Step 1: Relaxed Hand ]       [ Step 2: Fingers Straight ]  [ Step 3: Closed Fist ]  |
|  - Rest hand naturally          - Extend fingers 100% flat    - Close hand into fist   |
|  - Records resting flex         - Calibrates 0.0 flex min     - Calibrates 1.0 flex max|
|  - Calculates gyro offsets      - Record (3s)                 - Record (3s)            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                      [ Compute & Save Calibration Profile ]
                       Generates data/calibration/UXX.json
```

| Component Name | Element ID | Function |
|---|---|---|
| **User Selector** | `#calibUserSelect` | Chooses which user profile (`U01`–`U05`) to calibrate. |
| **Record Relaxed (3s)** | `#btnCalibRelaxed` | Captures 150 frames of neutral hand posture to compute gyro zero-rate offsets and resting baseline. |
| **Record Straight (3s)** | `#btnCalibStraight` | Captures 150 frames with fingers fully extended to define the minimum ADC voltage threshold ($0.0$). |
| **Record Bent (3s)** | `#btnCalibBent` | Captures 150 frames with fingers closed in a fist to define the maximum ADC voltage threshold ($1.0$). |
| **Save Profile Button** | `#btnSaveCalibration` | Computes per-sensor normalization slopes and saves `data/calibration/<user_id>.json`. |

---

### 5.7 Tab 5: Model Training Engine

Provides interactive training of both static and dynamic machine learning models.

| Component Name | Element ID | Function |
|---|---|---|
| **Data Splitting Strategy** | `#trainEvalModeSelect` | Toggles between: <br>• **User-Independent Split**: Trains on `U01`–`U04` and tests exclusively on unseen subject `U05` (evaluates generalization to new users). <br>• **Stratified Split**: 80/20 train/test split. |
| **Train Baseline Models** | `#btnTrainBaseline` | Concurrently trains **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**, calculates cross-validation metrics, saves models to `models/saved/`, and sets the best performing model active. |
| **Train Dynamic LSTM** | `#btnTrainLSTM` | Trains a 2-layer PyTorch Long Short-Term Memory network over 30-step sliding windows for continuous dynamic gestures (e.g. `HELLO` waving, `YES` nodding). |

---

### 5.8 Tab 6: Model Evaluation & Metrics

Presents empirical validation metrics to verify model accuracy and safety before real-world deployment.

| Component Name | Element ID | Description |
|---|---|---|
| **Model Comparison Table** | `#modelComparisonTableBody` | Side-by-side comparison of Decision Tree, Random Forest, and SVM displaying **Accuracy**, **Precision (Weighted)**, **Recall (Weighted)**, **F1-Score (Weighted)**, and **Training Execution Time**. Features a `★ BEST MODEL` badge for the top performer. |
| **Refresh Metrics Button** | Inline action | Reloads evaluation reports from `models/evaluation/model_comparison.json`. |
| **Normalized Confusion Matrix** | `#cmHeatmapContainer` | 20×20 interactive matrix styled like the **GitHub Contribution Graph**. Diagonal cells represent true positive rates color-coded in levels of GitHub Green (`#0e4429`, `#006d32`, `#26a641`, `#39d353`). Off-diagonal misclassifications are highlighted in warning red (`#da3633`). |
| **Per-Gesture Classification Report** | `#perClassReportContainer` | Comprehensive table listing Precision, Recall, F1-Score, and Support sample counts for each of the 20 ISL vocabulary signs. |

---

### 5.9 Tab 7: ISL Vocabulary Explorer

An interactive visual dictionary displaying all 20 signs supported by the system.

- **Vocabulary Grid (`#gesturesVocabularyGrid`)**: Dynamically populated grid of 20 cards.
- **Card Contents**:
  - Sign ID and number (`#1 HELLO`, `#2 THANK_YOU`, etc.).
  - English display word and Hindi Devanagari translation (`नमस्ते`).
  - Category badge: `Greetings`, `Basic Comm.`, `Emergency`, `Needs & Actions`, or `Pronouns`.
  - Anatomical posture description (e.g., `"Open hand, palm forward, waving side-to-side"`).
  - Sensor profile breakdown (expected flexion for thumb through little finger, plus IMU acceleration vectors).
  - **Interactive Click Action**: Clicking any card immediately sets that gesture in the live simulator, allowing the user to observe the 3D visualizer response.

---

### 5.10 Tab 8: Real-Time Translation History

Chronological audit log of all recognized gestures during an active session.

- **History Table (`#historyTableBody`)**:
  - **Timestamp**: Time of recognition (hh:mm:ss).
  - **English Sign**: Recognized gesture name with GitHub code styling.
  - **Hindi Translation**: Regional Devanagari script.
  - **Confidence**: Model probability percentage badge.
  - **Audio Replay (`🔊 Replay`)**: Interactive button allowing the user to replay voice synthesis for that specific past gesture on demand.

---

### 5.11 Tab 9: Hardware Settings & TinyML Exporter

Connects physical hardware and provides edge model compilation.

| Component Name | Element ID | Purpose |
|---|---|---|
| **COM / Bluetooth Port Dropdown** | `#serialPortSelect` | Lists all active serial and Bluetooth COM ports detected by the operating system. |
| **Refresh Ports Button** | `#btnRefreshPorts` | Queries `/api/serial/ports` to scan for newly plugged USB cables or paired Bluetooth devices. |
| **Connect Hardware Button** | `#btnConnectSerial` | Opens a serial stream with the ESP32 at 115,200 Baud, automatically handling packet framing and checksum verification. |
| **Bluetooth Setup Guide** | Static instructions | 3-step walkthrough for pairing `ISL-Glove-ESP32` in Windows Bluetooth settings. |
| **TinyML C++ Exporter Button** | `.btn-primary` (onclick: `exportTinyML()`) | Converts the active Scikit-Learn model into zero-dependency C++ code (`ISL_TinyML_Model.h`) for offline execution inside the ESP32 microcontroller (<1ms execution time). |
| **Export Result Banner** | `#tinymlExportResult` | Displays confirmation of exported header file path and total rule count. |

---

### 5.12 Global Notification System (Toasts)

- **Toast Container (`.toast-container`)**: Floating notifications rendered at the bottom-right corner of the screen.
- **Notification Types**:
  - `info`: General informational messages (blue border).
  - `success`: Calibration saved, models trained, dataset generated (green border).
  - `error`: Serial connection failure, invalid recording parameters (red border).
- Automatically dismisses after 4 seconds with a smooth fade-out animation.

---

## 6. Real-Time WebSocket Protocol & Telemetry Schema

The web client establishes a persistent connection to:
```
ws://127.0.0.1:8000/ws/stream
```

### Inbound Sensor Frame Payload (Server -> Client @ 50 Hz)

```json
{
  "type": "sensor_frame",
  "timestamp": 1729426210.45,
  "mode": "simulated",
  "raw_sample": {
    "timestamp": 1729426210.45,
    "user_id": "U01",
    "gesture": "HELLO",
    "thumb": 0.05,
    "index": 0.05,
    "middle": 0.05,
    "ring": 0.05,
    "little": 0.05,
    "acc_x": 0.002,
    "acc_y": 0.981,
    "acc_z": 0.152,
    "gyro_x": 0.051,
    "gyro_y": 0.452,
    "gyro_z": -0.103,
    "is_simulated": true
  },
  "prediction": {
    "gesture": "HELLO",
    "confidence": 0.954,
    "model_name": "svm",
    "top_k": [
      {"gesture": "HELLO", "probability": 0.954},
      {"gesture": "WELCOME", "probability": 0.031},
      {"gesture": "STOP", "probability": 0.015}
    ],
    "text_en": "Hello",
    "text_hi": "नमस्ते"
  },
  "is_recording": false
}
```

---

## 7. Speech Synthesis & Sentence Construction Engine

### 1. Temporal Debouncing
To prevent duplicate voice utterances while a user maintains a sign, `app.js` enforces a **2,500ms debounce threshold**:
```javascript
const now = Date.now();
if (now - this.lastSpokenTime > this.speechDebounceMs || this.lastSpokenGesture !== gesture) {
    this.speakText(text);
    this.lastSpokenGesture = gesture;
    this.lastSpokenTime = now;
}
```

### 2. Sentence Concatenation
Recognized gestures exceeding a **75% confidence threshold** are appended to the sentence buffer. Repeating consecutive signs are deduplicated, building natural conversational phrases (e.g., `"Hello"` + `"Welcome"` + `"Please"` -> `"Hello Welcome Please"`).

### 3. Native Multilingual Speech
Utilizes the client browser's `SpeechSynthesis` API:
- **English**: Voices tagged `en-US` or `en-IN`.
- **Hindi**: Voices tagged `hi-IN` (e.g., Google हिन्दी or Microsoft Hemant/Kalpana).

---

## 8. Setup, Execution & Verification

### Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Edge, Firefox)

### Step 1: Install Dependencies
```bash
pip install fastapi uvicorn websockets pyserial scikit-learn numpy pandas joblib torch httpx pytest
```

### Step 2: Run the Application Server
```bash
python backend/main.py
```
*(Or directly with Uvicorn: `uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`)*

### Step 3: Access the Application
Open your browser and navigate to:
- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger REST Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Step 4: Run Verification Tests
```bash
python run_tests.py
```
*(Executes all 16 end-to-end tests covering simulation, preprocessor scaling, calibration, ML training, prediction debouncing, LSTM dynamic sequences, and REST APIs).*
