#ifndef DISPLAY_HANDLER_H
#define DISPLAY_HANDLER_H

#include <Arduino.h>
#include <Wire.h>
#include "config.h"

class DisplayHandler {
private:
    bool _isAvailable;

public:
    DisplayHandler();
    bool begin();
    void showStatus(const String &state, const String &gesture, float confidence, bool btConnected);
    void showCalibrationStep(int step, const String &instruction);
    bool isAvailable() const { return _isAvailable; }
};

#endif // DISPLAY_HANDLER_H
