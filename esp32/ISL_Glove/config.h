
const int thumbPin = 32;
const int indexPin = 33;
const int middlePin = 34;
const int ringPin = 35;
const int littlePin = 39;

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  Serial.println(
    "thumb,index,middle,ring,little"
  );
}

void loop() {
  int thumb = analogRead(thumbPin);
  int index = analogRead(indexPin);
  int middle = analogRead(middlePin);
  int ring = analogRead(ringPin);
  int little = analogRead(littlePin);

  Serial.print(thumb);
  Serial.print(",");

  Serial.print(index);
  Serial.print(",");

  Serial.print(middle);
  Serial.print(",");

  Serial.print(ring);
  Serial.print(",");

  Serial.println(little);

  delay(2000);
}