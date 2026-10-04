# ISL Glove – Machine Learning Based Indian Sign Language Recognition Using a Wearable Sensor Glove

[![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142.2-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![ESP32](https://img.shields.io/badge/Hardware-ESP32%20%7C%20MPU6050%20%7C%20Flex-E7352C.svg)](https://www.espressif.com/)

An end-to-end wearable IoT and Machine Learning system for real-time **Indian Sign Language (ISL)** gesture recognition, translation, and speech synthesis. The system captures finger bending and hand spatial kinematics using a wearable glove instrumented with **5 Flex Sensors** and an **MPU6050 6-DOF IMU**, transmitting 11 numerical features to an intelligent ML pipeline with user-independent evaluation and multilingual Text-to-Speech (English & Hindi).

---

## 📋 Table of Contents
- [1. Project Overview & Architecture](#1-project-overview--architecture)
- [2. Hardware Specifications & Wiring Pinout](#2-hardware-specifications--wiring-pinout)
- [3. Predefined ISL Gesture Vocabulary (20 Signs)](#3-predefined-isl-gesture-vocabulary-20-signs)
- [4. Software Architecture & Directory Structure](#4-software-architecture--directory-structure)
- [5. Installation & Quick Start](#5-installation--quick-start)
- [6. Running the System (Simulator vs Hardware)](#6-running-the-system-simulator-vs-hardware)
- [7. Dataset Collection System & CSV Format](#7-dataset-collection-system--csv-format)
- [8. 3-Step Calibration & Normalization System](#8-3-step-calibration--normalization-system)
- [9. Machine Learning Pipeline & User-Independent Validation](#9-machine-learning-pipeline--user-independent-validation)
- [10. Advanced Dynamic Gesture Recognition (Time-Series LSTM)](#10-advanced-dynamic-gesture-recognition-time-series-lstm)
- [11. Multilingual Text-to-Speech (TTS)](#11-multilingual-text-to-speech-tts)
- [12. ESP32 Arduino Firmware Setup](#12-esp32-arduino-firmware-setup)
- [13. Running Tests & Verification](#13-running-tests--verification)
- [14. Troubleshooting & FAQ](#14-troubleshooting--faq)

---

## 1. Project Overview & Architecture

The ISL Glove translates non-verbal Indian Sign Language gestures into spoken and written words. 

```
+-------------------------------------------------------------------------------+
|                             WEARABLE SENSOR GLOVE                             |
|  [Thumb Flex] [Index Flex] [Middle Flex] [Ring Flex] [Little Flex] [MPU6050]  |
|                                     |                                         |
|                                     v                                         |
|                          ESP32 Microcontroller                                |
|                (12-bit ADC Filtering + 50Hz Packetization)                     |
+-------------------------------------------------------------------------------+
                                      |
                       Bluetooth SPP / USB Serial Stream
                                      v
+-------------------------------------------------------------------------------+
|                                PYTHON BACKEND                                 |
|  - Real-Time Serial Reader / Biomechanical Simulator Generator                |
|  - 3-Step Per-User Calibration Normalizer (Relaxed / Straight / Bent)         |
|  - Preprocessor & Feature Standardization (11 Numerical Features)             |
|  - ML Inference Engine (Decision Tree, Random Forest, SVM, PyTorch LSTM)       |
|  - Temporal Debouncing & Confidence Probability Filter                        |
|  - Multilingual Speech Translation Engine (English & Hindi)                   |
|  - 50Hz WebSocket Telemetry Server (FastAPI + Uvicorn)                        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                            MODERN WEB DASHBOARD                               |
|  - Real-Time 3D Hand Skeletal Posture Visualizer                              |
|  - Live Sensor Gauges & IMU Coordinate Sparklines                             |
|  - Gesture Recognition Hero Display with Hindi Script & Audio Speech          |
|  - Guided 3-Step Calibration Wizard                                           |
|  - Dataset Collection & Multi-User CSV Exporter                               |
|  - Interactive Confusion Matrix & Model Comparison Table                      |
+-------------------------------------------------------------------------------+
```

---

## 2. Hardware Specifications & Wiring Pinout

### Bill of Materials (BOM)
| Component | Quantity | Purpose |
|---|---|---|
| **ESP32 DevKit v1 (30 or 38-pin)** | 1 | Microcontroller with built-in Bluetooth Classic & WiFi |
| **2.2" or 4.5" Flex Sensors** | 5 | Measures flexion angle of Thumb, Index, Middle, Ring, Little fingers |
| **10kΩ Resistors (1/4W)** | 5 | Voltage divider resistors for flex sensors |
| **MPU-6050 Module** | 1 | 6-DOF Accelerometer ($\pm4g$) + Gyroscope ($\pm500^\circ/s$) |
| **0.96" I2C SSD1306 OLED (Optional)** | 1 | Real-time on-glove gesture display ($128 \times 64$) |
| **3.7V Li-Po Battery + TP4056 Charger** | 1 | Wearable power supply |
| **Glove & Wiring Harness** | 1 | Elastic fabric glove with flexible silicone wires |

### Circuit Wiring Pinout Table
| Sensor / Module | Sensor Pin | ESP32 GPIO Pin | Function |
|---|---|---|---|
| **Thumb Flex** | Divider Output | **GPIO 34** (ADC1_CH6) | Analog voltage measurement |
| **Index Flex** | Divider Output | **GPIO 35** (ADC1_CH7) | Analog voltage measurement |
| **Middle Flex** | Divider Output | **GPIO 32** (ADC1_CH4) | Analog voltage measurement |
| **Ring Flex** | Divider Output | **GPIO 33** (ADC1_CH5) | Analog voltage measurement |
| **Little Flex** | Divider Output | **GPIO 39** (ADC1_CH3 / VN) | Analog voltage measurement |
| **MPU6050** | SDA | **GPIO 21** | I2C Data Line |
| **MPU6050** | SCL | **GPIO 22** | I2C Clock Line |
| **MPU6050** | VCC / GND | 3.3V / GND | Power |
| **OLED Display** | SDA / SCL | **GPIO 21 / GPIO 22** | Shared I2C Bus (Addr: `0x3C`) |
| **Status LED** | Anode | **GPIO 2** | Built-in Blue Status Blink |
| **Boot Button** | Pin | **GPIO 0** | Interactive On-Glove Calibration |

> [!IMPORTANT]
> Always connect flex sensors to **ADC1 pins (GPIO 32–39)**. On ESP32, ADC2 cannot be used concurrently with Bluetooth/WiFi.

---

## 3. Predefined ISL Gesture Vocabulary (20 Signs)

The 20 predefined vocabulary gestures are defined in [`config/gestures.json`](file:///d:/ISL-%20Glove/config/gestures.json):

| # | Gesture ID | English Display | Hindi Translation | Category | Key Finger & IMU Posture |
|---|---|---|---|---|---|
| 1 | `HELLO` | Hello | नमस्ते | Greetings | Open hand, palm forward, waving side-to-side |
| 2 | `THANK_YOU` | Thank You | धन्यवाद | Greetings | Open flat hand sweeping forward from chin |
| 3 | `PLEASE` | Please | कृपया | Basic Comm. | Flat palm resting gently over chest |
| 4 | `YES` | Yes | हाँ | Basic Comm. | Closed fist nodding up and down ($+Z$ accel) |
| 5 | `NO` | No | नहीं | Basic Comm. | Index & middle closing onto thumb horizontally |
| 6 | `HELP` | Help | मदद | Emergency | Fist with thumb extended upward, lifted up |
| 7 | `GOOD` | Good | अच्छा | Basic Comm. | Thumbs up held steady ($+Z$ gravity) |
| 8 | `BAD` | Bad | बुरा | Basic Comm. | Thumbs down held steady ($-Z$ gravity) |
| 9 | `EAT` | Eat / Food | खाना | Needs & Actions | Fingertips bunched touching mouth |
| 10 | `DRINK` | Drink / Water | पानी पीना | Needs & Actions | C-hand tilted upwards |
| 11 | `I_LOVE_YOU` | I Love You | मुझे तुमसे प्यार है | Basic Comm. | Thumb, index, little extended; middle & ring bent |
| 12 | `PEACE` | Peace / Victory | शांति | Basic Comm. | Index & middle in V-shape; ring & little bent |
| 13 | `NAME` | Name | नाम | Basic Comm. | H-shape index and middle tapping |
| 14 | `ME` | Me / I | मैं | Pronouns | Index finger pointing towards chest |
| 15 | `YOU` | You | आप / तुम | Pronouns | Index finger pointing forward |
| 16 | `HOME` | Home | घर | Needs & Actions | Fingertips angled together in roof shape |
| 17 | `STOP` | Stop | रुको | Basic Comm. | Open flat hand held vertically forward |
| 18 | `BATHROOM` | Bathroom | शौचालय | Needs & Actions | T-handshape shaking side-to-side |
| 19 | `WELCOME` | Welcome | स्वागत है | Greetings | Open palm sweeping inward towards chest |
| 20 | `EMERGENCY` | Emergency | आपातकाल | Emergency | Open hand waving rapidly with high gyro rate |

---

## 4. Software Architecture & Directory Structure

```
d:/ISL- Glove/
├── backend/
│   ├── main.py                  # FastAPI REST + WebSocket Server
│   ├── simulator.py             # Biomechanical Hand Sensor Simulator (50Hz)
│   ├── calibration_service.py   # 3-step User Calibration & Normalization Engine
│   ├── prediction_service.py    # Real-Time ML Inference & Confidence Scoring
│   ├── serial_bluetooth.py      # Serial/Bluetooth Hardware COM Reader with Checksum
│   └── tts_service.py           # Multilingual Speech Synthesis & Sentence Builder
├── ml/
│   ├── preprocessor.py          # Data cleaning, scaling, and time-series windowing
│   ├── dataset_manager.py       # CSV dataset validation, user-independent splitting
│   ├── train_baseline.py        # Decision Tree, Random Forest, SVM training
│   ├── evaluate.py              # User-independent & stratified cross-validation
│   ├── model_selector.py        # Dynamic active model selector
│   └── dynamic_lstm.py          # Time-Series PyTorch LSTM Classifier
├── esp32/
│   ├── ISL_Glove.ino            # Main Arduino C++ sketch (50Hz state machine)
│   ├── config.h                 # Pinout, Baud rate, and protocol settings
│   ├── FlexSensors.h / .cpp     # Multi-sample ADC averaging & bounds check
│   ├── MPU6050_Handler.h / .cpp # I2C IMU reading, scaling, and offset zeroing
│   ├── Bluetooth_Handler.h / .cpp# Bluetooth Serial transmission & framing
│   └── Display_Handler.h / .cpp  # Optional SSD1306 OLED status display
├── frontend/
│   ├── index.html               # Modern Single Page Application UI
│   ├── css/style.css            # Cyber-Glassmorphism dark theme
│   ├── js/app.js                # App controller, Web Speech API, tab router
│   ├── js/visualizer.js         # Interactive Canvas 3D hand visualizer
│   └── js/charts.js             # Confusion matrix & telemetry metrics
├── config/
│   ├── gestures.json            # 20 ISL gestures definitions & profiles
│   ├── hardware_config.json     # Pinouts, sampling rate, protocol specs
│   └── app_config.json          # Ports, confidence thresholds, file paths
├── data/
│   ├── simulated/               # Simulated datasets (CSV)
│   ├── real/                    # Real sensor glove recordings (CSV)
│   └── calibration/             # Per-user calibration JSON profiles
├── models/
│   ├── saved/                   # Trained models (.joblib, .pth, scalers)
│   └── evaluation/              # Model comparison reports & confusion matrices
├── tests/
│   ├── test_simulator.py        # Unit tests for simulator
│   ├── test_preprocessor.py     # Unit tests for scaling & preprocessor
│   ├── test_calibration.py      # Unit tests for calibration computation
│   ├── test_ml_pipeline.py      # Unit tests for ML baseline trainers
│   ├── test_prediction.py       # Unit tests for real-time inference
│   └── test_api.py              # Integration tests for FastAPI endpoints
├── run_tests.py                 # Standalone test runner (100% verified)
└── README.md                    # Project documentation
```

---

## 5. Installation & Quick Start

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Modern web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox)

### Step 1: Install Dependencies
```bash
pip install fastapi uvicorn websockets pyserial scikit-learn numpy pandas joblib torch httpx pytest
```

### Step 2: Launch the Application Server
```bash
python backend/main.py
```
Or with Uvicorn directly:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Open the Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## 6. Running the System (Simulator vs Hardware)

The system features dual operating modes switchable with a single click in the top header:

### Mode 1: Simulated Development Mode (No Hardware Needed)
1. In the top-right header, ensure **"Simulator"** is active.
2. Go to the **Dashboard** or **Simulator Controller**.
3. Select any gesture (e.g. `HELLO`, `YES`, `GOOD`, `I_LOVE_YOU`) and user profile (`U01` to `U05`).
4. Watch the **3D Hand Visualizer** flex its fingers and tilt according to the biomechanical simulator at 50Hz, with live classification and voice speech!

### Mode 2: Real ESP32 Hardware Glove Mode
1. Pair your laptop with the Bluetooth device `ISL-Glove-ESP32` (or connect via USB cable).
2. Go to the **Settings** tab in the dashboard.
3. Select the ESP32 COM port and click **"Connect to Glove Hardware"**.
4. The system switches to real-time hardware telemetry streaming.

---

## 7. Dataset Collection System & CSV Format

### CSV Data Format
Both real and simulated datasets strictly adhere to the 14-column standard format:

```csv
timestamp,user_id,gesture,thumb,index,middle,ring,little,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,is_simulated
1727878390.12,U01,HELLO,0.05,0.05,0.05,0.05,0.05,0.002,0.981,0.152,0.051,0.452,-0.103,False
```

### Data Collection Workflow
1. Navigate to the **Dataset Collection** tab.
2. Select your **User ID** (e.g. `U01`, `U02`) and **Gesture** to record.
3. Select recording duration (3s, 5s, or 10s).
4. Perform the sign gesture and click **"⏺️ Start Recording"**.
5. Sensor samples are recorded at 50Hz, validated, and appended to CSV files in `data/real/` (or `data/simulated/`).
6. Click **"⚡ Generate Simulated Dataset"** anytime to populate 10,000 multi-user baseline samples across all 20 gestures.

---

## 8. 3-Step Calibration & Normalization System

Due to variations in hand sizes and sensor resistance drift, raw flex ADC readings must be normalized per user.

### Calibration Protocol
1. **Step 1: Relaxed Hand** – Hand rests naturally. Computes neutral gyro zero-rate bias offsets.
2. **Step 2: Fingers Straight** – All fingers extended flat. Sets the $0.0$ minimum bending limit ($V_{straight}$).
3. **Step 3: Fingers Bent (Fist)** – Hand closed in a tight fist. Sets the $1.0$ maximum bending limit ($V_{bent}$).

### Normalization Formula
$$\text{Flex}_{\text{norm}} = \text{clip}\left( \frac{\text{ADC}_{\text{raw}} - V_{\text{straight}}}{V_{\text{bent}} - V_{\text{straight}}}, 0.0, 1.0 \right)$$
$$\text{Gyro}_{\text{calib}} = \text{Gyro}_{\text{raw}} - \text{Bias}_{\text{offset}}$$

Profiles are stored in `data/calibration/<user_id>_calibration.json` and applied automatically during inference.

---

## 9. Machine Learning Pipeline & User-Independent Validation

### Evaluated Algorithms
1. **Decision Tree** (`DecisionTreeClassifier(max_depth=15, min_samples_split=4)`)
2. **Random Forest** (`RandomForestClassifier(n_estimators=100, max_depth=16)`)
3. **Support Vector Machine (SVM)** (`SVC(C=10.0, kernel='rbf')`)

### User-Independent Cross-Validation
To prove generalizability to unseen subjects, the system supports user-independent splits:
- **Training Set**: Users `U01`, `U02`, `U03`, `U04`
- **Testing Set**: User `U05` (unseen during training)

### Performance Metrics Output
The training pipeline computes:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Macro & Weighted)**: $\frac{TP}{TP + FP}$
- **Recall (Macro & Weighted)**: $\frac{TP}{TP + FN}$
- **F1-Score (Macro & Weighted)**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Normalized Confusion Matrix**: $20 \times 20$ True vs Predicted heatmap.

> [!NOTE]
> The system dynamically selects the active deployment model based on validation metrics rather than blindly assuming Random Forest is best.

---

## 10. Advanced Dynamic Gesture Recognition (Time-Series LSTM)

For moving gestures with kinematic trajectories (e.g. `HELLO` waving, `YES` nodding, `EMERGENCY` oscillation), the project provides an advanced deep learning sequential model:

- **Architecture**: 2-Layer PyTorch LSTM ($11 \to 64 \to 32 \to 20 \text{ classes}$) with Dropout ($0.2$)
- **Input Dimension**: $(N, 30, 11)$ sliding window tensors ($30 \text{ frames} \approx 600\text{ms}$)
- **Inference**: Rolling FIFO queue buffer with sequence classification.

---

## 11. Multilingual Text-to-Speech (TTS)

When a gesture is recognized with confidence $\ge 70\%$, the system synthesizes speech:
- **English**: "Hello", "Thank you", "Please", "Emergency", etc.
- **Hindi (हिंदी)**: "नमस्ते", "धन्यवाद", "कृपया", "आपातकाल", etc.
- **Sentence Builder**: Concatenates recognized gestures into fluent conversational phrases with one-click speech playback.

---

## 12. ESP32 Arduino Firmware Setup

### Flashing Instructions
1. Open the [Arduino IDE](https://www.arduino.cc/en/software).
2. Install the **ESP32 Board Package**:
   - `File -> Preferences -> Additional Boards Manager URLs`: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - `Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module`.
3. Open [`esp32/ISL_Glove.ino`](file:///d:/ISL-%20Glove/esp32/ISL_Glove.ino).
4. Verify pin numbers in [`esp32/config.h`](file:///d:/ISL-%20Glove/esp32/config.h).
5. Select Port and click **Upload** (Hold the `BOOT` button on ESP32 if upload stalls at `Connecting...`).

### Packet Framing Protocol
The glove transmits at 50Hz in standard NMEA-style framing:
```
$ISL,thumb,index,middle,ring,little,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z*CHECKSUM\n
```

---

## 13. Running Tests & Verification

The project includes an automated test suite verifying all 7 subsystems:

```bash
python run_tests.py
```

Expected output:
```
======================================================================
      ISL GLOVE - SYSTEM VERIFICATION & TEST SUITE
======================================================================

[SUITE 1] Testing Biomechanical Sensor Simulator...
  [PASS] Simulator Initialization & 20 Gestures Config
  [PASS] Single 11-Feature Sample Generation
  [PASS] Anatomical Finger Bending Bounds
  [PASS] Multi-User Simulated Dataset Generation

[SUITE 2] Testing Preprocessor & Feature Scaling...
  [PASS] DataFrame Cleaning, Validation & Fit-Transform
  [PASS] Single 11-Feature Sample Inference Scaling

[SUITE 3] Testing 3-Step Calibration & Per-User Normalization...
  [PASS] Default Calibration Profile Baseline
  [PASS] 3-Step Calibration (Relaxed, Straight, Bent) Computation

[SUITE 4] Testing ML Training (Decision Tree, Random Forest, SVM)...
  [PASS] Decision Tree vs Random Forest vs SVM Training & Evaluation

[SUITE 5] Testing Real-Time Prediction Engine & Confidence Scoring...
  [PASS] Real-Time Gesture Inference & Top-K Probability Output

[SUITE 6] Testing Advanced Time-Series PyTorch LSTM Model...
  [PASS] PyTorch Time-Series LSTM Training & Sequence Windowing

[SUITE 7] Testing FastAPI Endpoints...
  [PASS] GET /api/health
  [PASS] GET /api/gestures (20 ISL signs)
  [PASS] POST /api/mode (simulator/hardware)
  [PASS] POST /api/simulator/set_gesture
  [PASS] GET /api/serial/ports

======================================================================
TEST RESULTS: 16/16 PASSED (0 FAILED)
======================================================================
ALL TESTS PASSED SUCCESSFULLY! The system is 100% verified.
```

---

## 14. Troubleshooting & FAQ

### 1. How do I test the project without physical hardware?
The project runs out of the box in **Simulator Mode**. Start the backend (`python backend/main.py`), open `http://127.0.0.1:8000/`, and change gestures in the **Simulator Controller** or **Vocabulary** tab.

### 2. Why does the ESP32 upload fail with `Connecting... ___...___`?
Press and hold the physical **BOOT (GPIO0)** button on the ESP32 board while the Arduino IDE prints `Connecting...`, then release once the upload percentage starts.

### 3. How do I pair ESP32 Bluetooth with Windows?
1. Power the ESP32.
2. Open Windows **Settings $\to$ Bluetooth & Devices $\to$ Add Device**.
3. Select **`ISL-Glove-ESP32`**.
4. Open **Device Manager $\to$ Ports (COM & LPT)** to find the assigned outgoing COM port (e.g. `COM4`).
5. Select this COM port in the dashboard **Settings** tab and click **Connect**.

### 4. Can I add more gestures?
Yes! Simply append new gesture objects to [`config/gestures.json`](file:///d:/ISL-%20Glove/config/gestures.json) with their expected finger profiles and IMU orientations. The ML preprocessor, dataset generator, and UI will automatically adapt.
