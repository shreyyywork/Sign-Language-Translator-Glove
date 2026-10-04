#ifndef MPU6050_HANDLER_H
#define MPU6050_HANDLER_H

#include <Arduino.h>
#include <Wire.h>
#include "config.h"

struct IMUData {
    float acc_x;
    float acc_y;
    float acc_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
};

class MPU6050Handler {
private:
    uint8_t _addr;
    IMUData _offsets;
    bool _isConnected;

    void writeRegister(uint8_t reg, uint8_t val);
    void readRegisters(uint8_t reg, uint8_t count, uint8_t *dest);

public:
    MPU6050Handler(uint8_t addr = MPU6050_I2C_ADDR);
    bool begin();
    IMUData read();
    void calibrateOffsets();
    bool isConnected() const { return _isConnected; }
    bool isValid(const IMUData &data);
};

#endif // MPU6050_HANDLER_H
