const int flexPin = 32;

// EMA filter parameter
float alpha = 0.15;

// Filtered sensor value
float filteredValue = 0;

bool firstReading = true;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  delay(500);

  Serial.println("SignBridge - EMA Flex Sensor");
  Serial.println("Raw Value,Filtered Value");
}

void loop() {

  // Read raw ADC value
  int rawValue = analogRead(flexPin);

  // Initialize EMA with first reading
  if (firstReading) {
    filteredValue = rawValue;
    firstReading = false;
  }

  // EMA calculation
  filteredValue =
      alpha * rawValue +
      (1.0 - alpha) * filteredValue;

  // Display results
  Serial.print(rawValue);
  Serial.print(",");

  Serial.println(filteredValue, 2);

  delay(20);
}


