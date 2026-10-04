#ifndef BLUETOOTH_HANDLER_H
#define BLUETOOTH_HANDLER_H

#include <Arduino.h>
#include <BluetoothSerial.h>
#include "config.h"
#include "FlexSensors.h"
#include "MPU6050_Handler.h"

class BluetoothHandler {
private:
    BluetoothSerial _serialBT;
    bool _isInitialized;
    bool _isClientConnected;

    uint8_t calculateChecksum(const String &payload);

public:
    BluetoothHandler();
    bool begin(const char* deviceName = DEVICE_NAME);
    void sendSensorPacket(const FlexData &flex, const IMUData &imu);
    void sendMessage(const String &msg);
    bool hasCommand();
    String readCommand();
    bool isConnected();
};

#endif // BLUETOOTH_HANDLER_H
