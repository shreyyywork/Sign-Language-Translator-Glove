/**
 * ISL Wearable Sensor Glove - ESP32 Hardware Configuration Header
 * All GPIO pin assignments, I2C addresses, and hardware constants are defined here.
 * Modify pins here to adapt to different custom PCB or breadboard wiring.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ==========================================
// 1. BLUETOOTH & SERIAL CONFIGURATION
// ==========================================
#define DEVICE_NAME         "ISL-Glove-ESP32"
#define SERIAL_BAUD_RATE    115200
#define SAMPLING_FREQ_HZ    50          // 50 Hz = 20ms sample period
#define SAMPLE_PERIOD_MS    (1000 / SAMPLING_FREQ_HZ)

// Connection Interface Modes:
// COMM_MODE_USB  : Hardware USB Serial only (USB Cable / Port)
//                  - Saves 30%+ SRAM & Flash memory
//                  - Disables Bluetooth RF radio to save power
//                  - Frees ADC2 pins for additional sensors
// COMM_MODE_BT   : Bluetooth Serial (SPP) only (Wireless Virtual COM Port)
// COMM_MODE_DUAL : Both USB Serial Port and Bluetooth COM Port enabled simultaneously
#define COMM_MODE_USB       1
#define COMM_MODE_BT        2
#define COMM_MODE_DUAL      3

// Default connection mode on boot
#define DEFAULT_COMM_MODE   COMM_MODE_DUAL

// ==========================================
// 2. FLEX SENSOR GPIO PIN MAPPINGS (ADC1)
// Note: Use ADC1 pins (GPIO 32-39) on ESP32 when using WiFi/Bluetooth
// ==========================================
#define PIN_FLEX_THUMB      34          // ADC1_CH6
#define PIN_FLEX_INDEX      35          // ADC1_CH7
#define PIN_FLEX_MIDDLE     32          // ADC1_CH4
#define PIN_FLEX_RING       33          // ADC1_CH5
#define PIN_FLEX_LITTLE     39          // ADC1_CH3 (Sensor VN)

// ADC Configuration
#define ADC_RESOLUTION_BITS 12          // 12-bit ADC (0 - 4095)
#define ADC_MAX_VALUE       4095.0
#define ADC_NUM_SAMPLES     8           // Oversampling filter count
#define VOLTAGE_REF         3.3

// ==========================================
// 3. I2C CONFIGURATION (MPU6050 & OLED)
// ==========================================
#define PIN_I2C_SDA         21
#define PIN_I2C_SCL         22
#define MPU6050_I2C_ADDR    0x68
#define OLED_I2C_ADDR       0x3C

// ==========================================
// 4. USER INTERACTION & STATUS LED
// ==========================================
#define PIN_STATUS_LED      2           // Built-in blue LED
#define PIN_CALIB_BUTTON    0           // Built-in BOOT button (active LOW)

// ==========================================
// 5. PACKET PROTOCOL DEFINITIONS
// ==========================================
#define PROTOCOL_HEADER     "$ISL"
#define PROTOCOL_SEPARATOR  ","
#define PROTOCOL_FOOTER     "*"

#endif // CONFIG_H
