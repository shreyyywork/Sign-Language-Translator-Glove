#include "Bluetooth_Handler.h"

BluetoothHandler::BluetoothHandler() {
    _isInitialized = false;
    _isClientConnected = false;
    _commMode = DEFAULT_COMM_MODE;
}

bool BluetoothHandler::begin(const char* deviceName, int mode) {
    _commMode = mode;

    // If USB only mode, skip Bluetooth initialization to conserve RAM and power
    if (_commMode == COMM_MODE_USB) {
        _isInitialized = false;
        Serial.println("[COMM] Active Mode: USB Port Only (Hardware UART @ 115200 baud)");
        return true;
    }

    // Initialize Bluetooth Serial for COMM_MODE_BT or COMM_MODE_DUAL
    if (_serialBT.begin(deviceName)) {
        _isInitialized = true;
        if (_commMode == COMM_MODE_BT) {
            Serial.println("[COMM] Active Mode: Bluetooth COM Port Only");
        } else {
            Serial.println("[COMM] Active Mode: Dual Mode (USB Port + Bluetooth COM Port)");
        }
        return true;
    }

    // Fallback to USB mode if Bluetooth initialization fails
    _commMode = COMM_MODE_USB;
    _isInitialized = false;
    Serial.println("[COMM] Bluetooth init failed, falling back to USB Port only.");
    return false;
}

void BluetoothHandler::setMode(int mode) {
    _commMode = mode;
}

const char* BluetoothHandler::getModeString() const {
    switch (_commMode) {
        case COMM_MODE_USB: return "USB_PORT";
        case COMM_MODE_BT: return "BLUETOOTH_COM";
        case COMM_MODE_DUAL: return "DUAL_USB_BT";
        default: return "UNKNOWN";
    }
}

bool BluetoothHandler::isUsbActive() const {
    return (_commMode == COMM_MODE_USB || _commMode == COMM_MODE_DUAL);
}

bool BluetoothHandler::isBtActive() const {
    return (_commMode == COMM_MODE_BT || _commMode == COMM_MODE_DUAL) && _isInitialized;
}

uint8_t BluetoothHandler::calculateChecksum(const String &payload) {
    uint8_t crc = 0;
    for (size_t i = 0; i < payload.length(); i++) {
        crc ^= (uint8_t)payload[i];
    }
    return crc;
}

void BluetoothHandler::sendSensorPacket(const FlexData &flex, const IMUData &imu) {
    // Format: $ISL,thumb,index,middle,ring,little,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z*CHECKSUM\n
    String payload = "ISL," +
        String(flex.thumb, 1) + "," +
        String(flex.index, 1) + "," +
        String(flex.middle, 1) + "," +
        String(flex.ring, 1) + "," +
        String(flex.little, 1) + "," +
        String(imu.acc_x, 3) + "," +
        String(imu.acc_y, 3) + "," +
        String(imu.acc_z, 3) + "," +
        String(imu.gyro_x, 3) + "," +
        String(imu.gyro_y, 3) + "," +
        String(imu.gyro_z, 3);

    uint8_t crc = calculateChecksum(payload);
    String fullPacket = "$" + payload + "*" + String(crc, HEX) + "\n";

    // Transmit over Bluetooth if active
    if (isBtActive()) {
        _serialBT.print(fullPacket);
    }
    // Transmit over USB Serial if active
    if (isUsbActive()) {
        Serial.print(fullPacket);
    }
}

void BluetoothHandler::sendMessage(const String &msg) {
    String formatted = "$MSG," + msg + "\n";
    if (isBtActive()) {
        _serialBT.print(formatted);
    }
    if (isUsbActive()) {
        Serial.print(formatted);
    }
}

bool BluetoothHandler::hasCommand() {
    if (isBtActive() && _serialBT.available()) return true;
    if (isUsbActive() && Serial.available()) return true;
    return false;
}

String BluetoothHandler::readCommand() {
    String cmd = "";
    if (isBtActive() && _serialBT.available()) {
        cmd = _serialBT.readStringUntil('\n');
    } else if (isUsbActive() && Serial.available()) {
        cmd = Serial.readStringUntil('\n');
    }
    cmd.trim();
    return cmd;
}

bool BluetoothHandler::isConnected() {
    if (isBtActive()) {
        return _serialBT.hasClient();
    }
    // In USB mode, Serial is always active once plugged into PC USB port
    return isUsbActive();
}
