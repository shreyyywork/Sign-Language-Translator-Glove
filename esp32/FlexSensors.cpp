#include "FlexSensors.h"

FlexSensors::FlexSensors() {
    _pins[0] = PIN_FLEX_THUMB;
    _pins[1] = PIN_FLEX_INDEX;
    _pins[2] = PIN_FLEX_MIDDLE;
    _pins[3] = PIN_FLEX_RING;
    _pins[4] = PIN_FLEX_LITTLE;
    _isCalibrated = false;

    // Default baseline values
    for (int i = 0; i < 5; i++) {
        _calib.straight[i] = 1200.0f;
        _calib.bent[i] = 3200.0f;
    }
}

void FlexSensors::begin() {
    analogReadResolution(ADC_RESOLUTION_BITS);
    // Attenuation 11dB allows reading up to ~3.3V
    analogSetAttenuation(ADC_11db);

    for (int i = 0; i < 5; i++) {
        pinMode(_pins[i], INPUT);
    }
}

float FlexSensors::readFilteredADC(uint8_t pin) {
    uint32_t sum = 0;
    for (int i = 0; i < ADC_NUM_SAMPLES; i++) {
        sum += analogRead(pin);
        delayMicroseconds(100);
    }
    return (float)sum / (float)ADC_NUM_SAMPLES;
}

FlexData FlexSensors::readRaw() {
    FlexData data;
    data.thumb  = readFilteredADC(_pins[0]);
    data.index  = readFilteredADC(_pins[1]);
    data.middle = readFilteredADC(_pins[2]);
    data.ring   = readFilteredADC(_pins[3]);
    data.little = readFilteredADC(_pins[4]);
    return data;
}

FlexData FlexSensors::readNormalized() {
    FlexData raw = readRaw();
    FlexData norm;

    float* raw_arr = (float*)&raw;
    float* norm_arr = (float*)&norm;

    for (int i = 0; i < 5; i++) {
        float span = _calib.bent[i] - _calib.straight[i];
        if (abs(span) < 1.0f) span = 1.0f;

        float val = (raw_arr[i] - _calib.straight[i]) / span;
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        norm_arr[i] = val;
    }

    return norm;
}

void FlexSensors::setCalibration(const CalibrationLimits &calib) {
    _calib = calib;
    _isCalibrated = true;
}

void FlexSensors::calibrateStraight() {
    for (int i = 0; i < 5; i++) {
        _calib.straight[i] = readFilteredADC(_pins[i]);
    }
}

void FlexSensors::calibrateBent() {
    for (int i = 0; i < 5; i++) {
        _calib.bent[i] = readFilteredADC(_pins[i]);
    }
    _isCalibrated = true;
}

bool FlexSensors::isValid(const FlexData &data) {
    const float* arr = (const float*)&data;
    for (int i = 0; i < 5; i++) {
        // Disconnected or shorted pin check
        if (isnan(arr[i]) || arr[i] < 0.0f || arr[i] > 4095.0f) {
            return false;
        }
    }
    return true;
}
