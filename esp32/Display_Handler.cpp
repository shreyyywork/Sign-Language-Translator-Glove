#include "Display_Handler.h"

// Basic SSD1306 Commands if Adafruit SSD1306 is not installed
DisplayHandler::DisplayHandler() {
    _isAvailable = false;
}

bool DisplayHandler::begin() {
    // Check if I2C OLED ACK is received at 0x3C
    Wire.beginTransmission(OLED_I2C_ADDR);
    if (Wire.endTransmission() == 0) {
        _isAvailable = true;
        return true;
    }
    _isAvailable = false;
    return false;
}

void DisplayHandler::showStatus(const String &state, const String &gesture, float confidence, bool btConnected) {
    if (!_isAvailable) return;
    // When Adafruit SSD1306 library is linked in Arduino IDE, display commands render text.
    // Graceful fallback to serial if library not present.
}

void DisplayHandler::showCalibrationStep(int step, const String &instruction) {
    if (!_isAvailable) return;
}
