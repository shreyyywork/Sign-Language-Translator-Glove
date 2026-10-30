/**
 * ISL Glove – Machine Learning Based Indian Sign Language Recognition Using a Wearable Sensor Glove
 * Main ESP32 Firmware Sketch
 *
 * Hardware Components:
 *   - ESP32 NodeMCU / DevKit v1
 *   - 5 Flex Sensors (Thumb: GPIO34, Index: GPIO35, Middle: GPIO32, Ring: GPIO33, Little: GPIO39)
 *   - MPU6050 6-DOF IMU (I2C SDA: GPIO21, SCL: GPIO22)
 *   - Status LED (GPIO2)
 *   - Calibration / Boot Button (GPIO0)
 *   - Optional 0.96" I2C SSD1306 OLED (0x3C)
 *
 * Features:
 *   - 50Hz (20ms) non-blocking sampling loop
 *   - Robust frame protocol: $ISL,thumb,index,middle,ring,little,ax,ay,az,gx,gy,gz*CRC\n
 *   - Bluetooth Serial & USB Serial streaming
 *   - Interactive calibration mode for user customization
 *   - Real-time sensor health validation
 */

#include "config.h"
#include "FlexSensors.h"
#include "MPU6050_Handler.h"
#include "Bluetooth_Handler.h"
#include "Display_Handler.h"

// System States
enum SystemState {
    STATE_IDLE,
    STATE_STREAMING,
    STATE_CALIBRATING,
    STATE_ERROR
};

// Global Handler Instances
FlexSensors flexSensors;
MPU6050Handler mpuHandler;
BluetoothHandler btHandler;
DisplayHandler displayHandler;

SystemState currentState = STATE_STREAMING;
unsigned long lastSampleTime = 0;
unsigned long lastStatusBlink = 0;
bool ledState = false;

// Calibration State Variables
int calibrationStep = 0;
unsigned long calibStepStart = 0;

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(500);
    Serial.println("\n=============================================");
    Serial.println("   ISL Glove - ESP32 Firmware Starting...   ");
    Serial.println("=============================================");

    pinMode(PIN_STATUS_LED, OUTPUT);
    pinMode(PIN_CALIB_BUTTON, INPUT_PULLUP);
    digitalWrite(PIN_STATUS_LED, HIGH);

    // Initialize Flex Sensors
    Serial.print("[INFO] Initializing 5 Flex Sensors... ");
    flexSensors.begin();
    Serial.println("OK");

    // Initialize MPU6050 IMU
    Serial.print("[INFO] Initializing MPU6050 IMU (I2C 0x68)... ");
    if (mpuHandler.begin()) {
        Serial.println("OK (Sensor Detected)");
    } else {
        Serial.println("WARNING: MPU6050 not found or connection failed. Using fallback values.");
    }

    // Initialize Communication Interface (USB Port / Bluetooth COM Port / Dual Mode)
    Serial.print("[INFO] Initializing Communication (Mode: ");
    Serial.print(DEFAULT_COMM_MODE == COMM_MODE_USB ? "USB Port Only" :
                 (DEFAULT_COMM_MODE == COMM_MODE_BT ? "Bluetooth COM Port Only" : "Dual USB + Bluetooth"));
    Serial.print(")... ");
    if (btHandler.begin(DEVICE_NAME, DEFAULT_COMM_MODE)) {
        Serial.println("OK");
    } else {
        Serial.println("WARNING: Falling back to USB Serial only");
    }

    // Initialize Optional OLED Display
    Serial.print("[INFO] Checking for OLED Display... ");
    if (displayHandler.begin()) {
        Serial.println("OK (0.96\" SSD1306 detected)");
    } else {
        Serial.println("None detected (Optional)");
    }

    digitalWrite(PIN_STATUS_LED, LOW);
    Serial.println("[INFO] Initialization Complete. Ready to stream at 50Hz.");
}

void handleCommands() {
    if (!btHandler.hasCommand()) return;

    String cmd = btHandler.readCommand();
    Serial.print("[CMD RECEIVED] ");
    Serial.println(cmd);

    if (cmd == "START" || cmd == "STREAM") {
        currentState = STATE_STREAMING;
        btHandler.sendMessage("STREAM_STARTED");
    } else if (cmd == "STOP" || cmd == "PAUSE") {
        currentState = STATE_IDLE;
        btHandler.sendMessage("STREAM_PAUSED");
    } else if (cmd == "CALIB_STRAIGHT") {
        flexSensors.calibrateStraight();
        btHandler.sendMessage("CALIB_STRAIGHT_SAVED");
    } else if (cmd == "CALIB_BENT") {
        flexSensors.calibrateBent();
        btHandler.sendMessage("CALIB_BENT_SAVED");
    } else if (cmd == "CALIB_IMU") {
        mpuHandler.calibrateOffsets();
        btHandler.sendMessage("CALIB_IMU_SAVED");
    } else if (cmd == "PING") {
        btHandler.sendMessage("PONG");
    } else if (cmd == "MODE_USB" || cmd == "SET_COMM_USB") {
        btHandler.setMode(COMM_MODE_USB);
        btHandler.sendMessage("COMM_MODE_SET:USB_PORT");
        Serial.println("[COMM] Switched to USB Port only mode.");
    } else if (cmd == "MODE_BT" || cmd == "SET_COMM_BT") {
        btHandler.setMode(COMM_MODE_BT);
        btHandler.sendMessage("COMM_MODE_SET:BLUETOOTH_COM");
        Serial.println("[COMM] Switched to Bluetooth COM Port only mode.");
    } else if (cmd == "MODE_DUAL" || cmd == "SET_COMM_DUAL") {
        btHandler.setMode(COMM_MODE_DUAL);
        btHandler.sendMessage("COMM_MODE_SET:DUAL_USB_BT");
        Serial.println("[COMM] Switched to Dual USB + Bluetooth mode.");
    } else if (cmd == "STATUS") {
        String st = "STATUS: State=" + String(currentState) +
                    ", Comm=" + String(btHandler.getModeString()) +
                    ", USB=" + String(btHandler.isUsbActive() ? "ACTIVE" : "OFF") +
                    ", BT=" + String(btHandler.isBtActive() ? (btHandler.isConnected() ? "CONNECTED" : "WAITING") : "OFF") +
                    ", MPU=" + String(mpuHandler.isConnected() ? "OK" : "DISCONNECTED");
        btHandler.sendMessage(st);
    }
}

void handlePhysicalButton() {
    // Check if BOOT button pressed to trigger on-device calibration
    if (digitalRead(PIN_CALIB_BUTTON) == LOW) {
        delay(50); // Debounce
        if (digitalRead(PIN_CALIB_BUTTON) == LOW) {
            Serial.println("[BTN] Calibration button pressed.");
            currentState = STATE_CALIBRATING;
            calibrationStep = 1;
            calibStepStart = millis();
            btHandler.sendMessage("CALIBRATION_STEP_1_STRAIGHT");
            digitalWrite(PIN_STATUS_LED, HIGH);

            // Wait for release
            while (digitalRead(PIN_CALIB_BUTTON) == LOW) {
                delay(10);
            }
        }
    }
}

void processCalibration() {
    // Step 1: Hold fingers straight for 3 seconds
    if (calibrationStep == 1) {
        if (millis() - calibStepStart >= 3000) {
            flexSensors.calibrateStraight();
            mpuHandler.calibrateOffsets();
            calibrationStep = 2;
            calibStepStart = millis();
            btHandler.sendMessage("CALIBRATION_STEP_2_BENT");
            Serial.println("[CALIB] Step 1 Done. Now close fingers into fist...");
        }
    }
    // Step 2: Hold fingers bent for 3 seconds
    else if (calibrationStep == 2) {
        if (millis() - calibStepStart >= 3000) {
            flexSensors.calibrateBent();
            calibrationStep = 0;
            currentState = STATE_STREAMING;
            btHandler.sendMessage("CALIBRATION_COMPLETE");
            Serial.println("[CALIB] Calibration Complete! Returning to streaming.");
            digitalWrite(PIN_STATUS_LED, LOW);
        }
    }
}

void loop() {
    handleCommands();
    handlePhysicalButton();

    if (currentState == STATE_CALIBRATING) {
        processCalibration();
    }

    unsigned long currentMillis = millis();

    // 50Hz Sample loop (every 20ms)
    if (currentMillis - lastSampleTime >= SAMPLE_PERIOD_MS) {
        lastSampleTime = currentMillis;

        // Read sensors
        FlexData flex = flexSensors.readRaw();
        IMUData imu = mpuHandler.read();

        // Validate sensors
        bool flexOk = flexSensors.isValid(flex);
        bool imuOk = mpuHandler.isValid(imu);

        if (currentState == STATE_STREAMING) {
            btHandler.sendSensorPacket(flex, imu);
        }

        // Heartbeat LED blink (1Hz)
        if (currentMillis - lastStatusBlink >= 500) {
            lastStatusBlink = currentMillis;
            ledState = !ledState;
            if (currentState == STATE_STREAMING) {
                digitalWrite(PIN_STATUS_LED, ledState ? HIGH : LOW);
            }
        }
    }
}
