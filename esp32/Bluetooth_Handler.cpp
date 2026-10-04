#include "Bluetooth_Handler.h"

BluetoothHandler::BluetoothHandler() {
    _isInitialized = false;
    _isClientConnected = false;
}

bool BluetoothHandler::begin(const char* deviceName) {
    if (_serialBT.begin(deviceName)) {
        _isInitialized = true;
        return true;
    }
    return false;
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

    // Transmit over both Bluetooth and Serial Monitor
    if (_isInitialized) {
        _serialBT.print(fullPacket);
    }
    Serial.print(fullPacket);
}

void BluetoothHandler::sendMessage(const String &msg) {
    String formatted = "$MSG," + msg + "\n";
    if (_isInitialized) {
        _serialBT.print(formatted);
    }
    Serial.print(formatted);
}

bool BluetoothHandler::hasCommand() {
    if (_isInitialized && _serialBT.available()) return true;
    if (Serial.available()) return true;
    return false;
}

String BluetoothHandler::readCommand() {
    String cmd = "";
    if (_isInitialized && _serialBT.available()) {
        cmd = _serialBT.readStringUntil('\n');
    } else if (Serial.available()) {
        cmd = Serial.readStringUntil('\n');
    }
    cmd.trim();
    return cmd;
}

bool BluetoothHandler::isConnected() {
    if (_isInitialized) {
        return _serialBT.hasClient();
    }
    return false;
}
