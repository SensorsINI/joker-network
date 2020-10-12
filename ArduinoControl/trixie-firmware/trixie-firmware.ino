
// finger control for trixsie robot
// last modification tobi Oct 2020

// IRF 510 power TO220AB package MOSFET has pin order from left facing device GDS

// use Arduino settings Arduino Nano, processor AtMega328P (old bootloader) for the chinese nanos with CH340 USB serial port.
// Note serial baud rate is 115200 baud for serial monitor. You might need to "sudo chmod a+rw /dev/ttyUSB0" on linux after each boot.
// To test with serial monitor, set line endings to "none" and enter commands 0, 1 in input field.

const String VERSION = "*** Trixsie Oct 2020 V1.0";
const String HELP = "Send char '1' to activate solenoid finger, '0' to relax it";

const int fingerPin = 3;  // setting this high will activate solenoide finger. But it should NOT be left high long, or the solenoide can burn out
const int butPin = 8; // pushing button will pull pin low (pin is configured input pullup with button tied to ground on other side of button)

const unsigned long PULSE_TIME_MS = 50; // pulse time in ms to drive finger out
const int  HOLD_DUTY_CYCLE = 10; // duty cycle for PWM output to hold finger out, range 0-255 for analogWrite
const char CMD_ACTIVATE_FINGER = '1', CMD_RELAX_FINGER = '0'; // python sends character '1' to activate finger, '0' to relax it
const int STATE_IDLE = 0, STATE_FINGER_PUSHING_OUT = 1, STATE_FINGER_HOLDING = 2;

const bool DEBUG = true;

unsigned long fingerActivatedTime = 0;
int state = 0, previousState = state, previousButState = HIGH;

void setup()
{

  Serial.begin(115200);  // initialize serial communications at max speed possible 115kbaud bps
  Serial.println(VERSION);
  Serial.print("Compile date and time: ");
  Serial.print(__DATE__);
  Serial.print(" ");
  Serial.println(__TIME__);
  if (DEBUG) Serial.println("Compiled with DEBUG=true"); else Serial.println("Compiled with DEBUG=false");
  Serial.print("Finger pulse time in ms: ");
  Serial.println(PULSE_TIME_MS);
  Serial.print("Finger hold duty cycle of 255: ");
  Serial.println(HOLD_DUTY_CYCLE);
  Serial.println(HELP);
  pinMode(fingerPin, OUTPUT);
  pinMode(butPin, INPUT_PULLUP);

}

void loop()
{
  if (Serial.available()) {  // process command char if we get one
    char c = Serial.read();
    switch (c) {
      case CMD_ACTIVATE_FINGER:
        state = STATE_FINGER_PUSHING_OUT;
        break;
      case CMD_RELAX_FINGER:
        state = STATE_IDLE;
        break;
      default:
        Serial.print("unknown command character recieved: ");
        Serial.println(c);
    }
  }

  bool but=digitalRead(butPin);
  
  if (but == LOW) { // button pressed
    if (previousButState == HIGH) {
      if (DEBUG) Serial.println("button pressed");
      state = STATE_FINGER_PUSHING_OUT;
    }
  } else if (but == HIGH) { // button pressed
    if (previousButState == LOW) {
      if (DEBUG) Serial.println("button released");
      state = STATE_IDLE;
    }
  }

  switch (state) {
    case STATE_IDLE:
      if (previousState != STATE_IDLE) {
        if (DEBUG) Serial.println("relaxing finger");
      }
      analogWrite(fingerPin, 0);
      break;
    case STATE_FINGER_PUSHING_OUT:
      if (previousState == STATE_IDLE) {
        fingerActivatedTime = millis();
        if (DEBUG) Serial.println("pushing finger out");
        analogWrite(fingerPin, 255);
      } else if (millis() - fingerActivatedTime > PULSE_TIME_MS) {
        state = STATE_FINGER_HOLDING;
        if (DEBUG) Serial.println("now holding finger out");
      }
      break;
    case STATE_FINGER_HOLDING:
      analogWrite(fingerPin, HOLD_DUTY_CYCLE);
      break;
  }
  previousState = state;
  previousButState=but;

}
