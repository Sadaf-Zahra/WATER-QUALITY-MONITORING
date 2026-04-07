// ---------------- Pins ----------------
#define PH_PIN A0
#define TURBIDITY_PIN A2
#define TEMP_PIN 2
#define TDS_PIN A1
#define LED_PIN 6

// ---------------- Thresholds ----------------
#define PH_MIN 6.5
#define PH_MAX 8.5

#define TURB_CLEAR_ON 650
#define TURB_DIRTY_ON 550

// -------- pH calibration --------
float calibration_value = 21.34;

bool waterClear = false;

// -------- pH buffer --------
int phBuffer[10];
int temp;

void setup() {
  Serial.begin(9600);

  pinMode(TEMP_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(LED_PIN, HIGH);
  delay(300);
  digitalWrite(LED_PIN, LOW);
}

// -------- Averaged analog read (Turbidity & TDS) --------
int readAveragedAnalog(int pin) {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
    delayMicroseconds(200);
  }
  return sum / 10;
}

// -------- Accurate pH Read Function (FROM YOUR WORKING CODE) --------
float readPH() {

  for (int i = 0; i < 10; i++) {
    phBuffer[i] = analogRead(PH_PIN);
    delay(30);
  }

  // sort values
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 10; j++) {
      if (phBuffer[i] > phBuffer[j]) {
        temp = phBuffer[i];
        phBuffer[i] = phBuffer[j];
        phBuffer[j] = temp;
      }
    }
  }

  unsigned long avgval = 0;
  for (int i = 2; i < 8; i++) {
    avgval += phBuffer[i];
  }

  float voltage = (float)avgval * 5.0 / 1024.0 / 6.0;
  float phValue = -5.70 * voltage + calibration_value;

  // safety clamp
  // if (phValue < 0)  phValue = 0;
  // if (phValue > 14) phValue = 14;

  return phValue;
}

void loop() {

  // -------- Read Sensors --------
  float phValue      = readPH();                 
  int turbidityRaw   = readAveragedAnalog(TURBIDITY_PIN);
  int tdsRaw         = readAveragedAnalog(TDS_PIN);
  int tempState      = digitalRead(TEMP_PIN);

  // -------- Turbidity (hysteresis) --------
  if (turbidityRaw > TURB_CLEAR_ON) waterClear = true;
  if (turbidityRaw < TURB_DIRTY_ON) waterClear = false;

  // -------- TDS --------
  float tdsValue = map(tdsRaw, 0, 1023, 0, 1200);

  // -------- Temperature --------
  float temperature = (tempState == LOW) ? 25.0 : 45.0;

  // -------- Status --------
  bool safe =
    (phValue >= PH_MIN && phValue <= PH_MAX) &&
    waterClear &&
    (tdsValue < 500) &&
    (tempState == LOW);

  digitalWrite(LED_PIN, safe ? HIGH : LOW);

  // -------- JSON --------
  Serial.print("{\"ph\":");
  Serial.print(phValue, 2);
  Serial.print(",\"turbidity\":");
  Serial.print(turbidityRaw);
  Serial.print(",\"turb\":\"");
  Serial.print(waterClear ? "CLEAR" : "DIRTY");
  Serial.print("\",\"temp\":");
  Serial.print(temperature, 1);
  Serial.print(",\"tds\":");
  Serial.print(tdsValue, 0);
  Serial.print(",\"status\":\"");
  Serial.print(safe ? "SAFE" : "UNSAFE");
  Serial.println("\"}");

  delay(500);
}
