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
    int _commMode;

    uint8_t calculateChecksum(const String &payload);

public:
    BluetoothHandler();
    bool begin(const char* deviceName = DEVICE_NAME, int mode = DEFAULT_COMM_MODE);
    void setMode(int mode);
    int getMode() const { return _commMode; }
    const char* getModeString() const;
    bool isUsbActive() const;
    bool isBtActive() const;
    void sendSensorPacket(const FlexData &flex, const IMUData &imu);
    void sendMessage(const String &msg);
    bool hasCommand();
    String readCommand();
    bool isConnected();
};

#endif // BLUETOOTH_HANDLER_H
