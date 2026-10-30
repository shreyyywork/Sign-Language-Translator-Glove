#ifndef FLEX_SENSORS_H
#define FLEX_SENSORS_H

#include <Arduino.h>
#include "config.h"

struct FlexData {
    float thumb;
    float index;
    float middle;
    float ring;
    float little;
};

struct CalibrationLimits {
    float straight[5];
    float bent[5];
};

class FlexSensors {
private:
    uint8_t _pins[5];
    CalibrationLimits _calib;
    bool _isCalibrated;

    float readFilteredADC(uint8_t pin);

public:
    FlexSensors();
    void begin();
    FlexData readRaw();
    FlexData readNormalized();
    void setCalibration(const CalibrationLimits &calib);
    void calibrateStraight();
    void calibrateBent();
    bool isValid(const FlexData &data);
};

#endif // FLEX_SENSORS_H
