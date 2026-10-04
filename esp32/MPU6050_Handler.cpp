#include "MPU6050_Handler.h"

// MPU6050 Register Map
#define MPU_PWR_MGMT_1    0x6B
#define MPU_SMPLRT_DIV    0x19
#define MPU_CONFIG        0x1A
#define MPU_GYRO_CONFIG   0x1B
#define MPU_ACCEL_CONFIG  0x1C
#define MPU_ACCEL_XOUT_H  0x3B
#define MPU_WHO_AM_I      0x75

MPU6050Handler::MPU6050Handler(uint8_t addr) {
    _addr = addr;
    _isConnected = false;
    _offsets = {0, 0, 0, 0, 0, 0};
}

void MPU6050Handler::writeRegister(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(_addr);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

void MPU6050Handler::readRegisters(uint8_t reg, uint8_t count, uint8_t *dest) {
    Wire.beginTransmission(_addr);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(_addr, count);
    for (uint8_t i = 0; i < count && Wire.available(); i++) {
        dest[i] = Wire.read();
    }
}

bool MPU6050Handler::begin() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // 400kHz Fast I2C

    // Verify connection via WHO_AM_I register (expected 0x68)
    uint8_t whoami = 0;
    readRegisters(MPU_WHO_AM_I, 1, &whoami);
    if (whoami != 0x68 && whoami != 0x70 && whoami != 0x72) {
        _isConnected = false;
        return false;
    }

    // Wake up MPU6050 (clear sleep bit)
    writeRegister(MPU_PWR_MGMT_1, 0x01); // Clock source: PLL with X axis gyro
    delay(10);

    // Set sample rate divider to 4 (1kHz / (1+4) = 200Hz internal)
    writeRegister(MPU_SMPLRT_DIV, 0x04);

    // Configure DLPF (Digital Low Pass Filter) ~ 44Hz bandwidth
    writeRegister(MPU_CONFIG, 0x03);

    // Gyro Full Scale: ±500 deg/s (scale = 65.5 LSB/(deg/s))
    writeRegister(MPU_GYRO_CONFIG, 0x08);

    // Accel Full Scale: ±4g (scale = 8192 LSB/g)
    writeRegister(MPU_ACCEL_CONFIG, 0x08);

    _isConnected = true;
    return true;
}

IMUData MPU6050Handler::read() {
    IMUData data = {0, 0, 0, 0, 0, 0};
    if (!_isConnected) return data;

    uint8_t buf[14];
    readRegisters(MPU_ACCEL_XOUT_H, 14, buf);

    int16_t raw_ax = (int16_t)((buf[0] << 8) | buf[1]);
    int16_t raw_ay = (int16_t)((buf[2] << 8) | buf[3]);
    int16_t raw_az = (int16_t)((buf[4] << 8) | buf[5]);
    // buf[6], buf[7] is Temperature
    int16_t raw_gx = (int16_t)((buf[8] << 8) | buf[9]);
    int16_t raw_gy = (int16_t)((buf[10] << 8) | buf[11]);
    int16_t raw_gz = (int16_t)((buf[12] << 8) | buf[13]);

    // Convert to standard units: Accel in g (±4g -> 8192 LSB/g), Gyro in rad/s (±500 deg/s -> 65.5 LSB/dps * (PI/180))
    data.acc_x = ((float)raw_ax / 8192.0f) - _offsets.acc_x;
    data.acc_y = ((float)raw_ay / 8192.0f) - _offsets.acc_y;
    data.acc_z = ((float)raw_az / 8192.0f) - _offsets.acc_z;

    float deg_to_rad = 0.0174532925f;
    data.gyro_x = (((float)raw_gx / 65.5f) * deg_to_rad) - _offsets.gyro_x;
    data.gyro_y = (((float)raw_gy / 65.5f) * deg_to_rad) - _offsets.gyro_y;
    data.gyro_z = (((float)raw_gz / 65.5f) * deg_to_rad) - _offsets.gyro_z;

    return data;
}

void MPU6050Handler::calibrateOffsets() {
    if (!_isConnected) return;

    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    const int samples = 100;

    for (int i = 0; i < samples; i++) {
        IMUData cur = read();
        sum_gx += cur.gyro_x;
        sum_gy += cur.gyro_y;
        sum_gz += cur.gyro_z;
        delay(5);
    }

    _offsets.gyro_x = sum_gx / samples;
    _offsets.gyro_y = sum_gy / samples;
    _offsets.gyro_z = sum_gz / samples;
}

bool MPU6050Handler::isValid(const IMUData &data) {
    if (isnan(data.acc_x) || isnan(data.acc_y) || isnan(data.acc_z) ||
        isnan(data.gyro_x) || isnan(data.gyro_y) || isnan(data.gyro_z)) {
        return false;
    }
    // Acceleration should be within physical limits (e.g. < 5g)
    float total_g = sqrt(data.acc_x*data.acc_x + data.acc_y*data.acc_y + data.acc_z*data.acc_z);
    return (total_g > 0.1f && total_g < 5.0f);
}
