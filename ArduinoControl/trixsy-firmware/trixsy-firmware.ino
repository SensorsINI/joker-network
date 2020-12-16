
// finger control for trixsie robot
// last modification tobi Oct 2020

#include <Watchdog.h> // https://github.com/janelia-arduino/Watchdog version 2.2.0 (not other watchdog library)
// install from library manager sketch/install library/library manager... or ctl-shift-i
#include <EEPROM.h>


// IRF 510 power TO220AB package MOSFET has pin order from left facing device GDS
// 2N3906 has pins in order EBC
// arduino schematic https://www.arduino.cc/en/uploads/Main/Arduino_Nano-Rev3.2-SCH.pdf

// use Arduino settings Arduino Nano, processor AtMega328P (old bootloader) for the chinese nanos with CH340 USB serial port.
// Note serial baud rate is 115200 baud for serial monitor. You might need to "sudo chmod a+rw /dev/ttyUSB0" on linux after each boot.
// To test with serial monitor, set line endings to "none" and enter commands 0, 1 in input field.
// or use minicom with  minicom -b 115200 -D /dev/ttyUSB0 -w 

const bool DEBUG = false; // set true to enable prints to serial port, but will slow down latency of response

Watchdog watchdog;
const String VERSION = "*** Trixsie Oct 2020 V1.0";
const String HELP = "Send char '1' to activate solenoid finger, '0' to relax it\n+/- increase/decrease pulse time, ]/[ increase/decrease hold duty cycle";

const int FINGER_PIN = 3;  // setting this high will activate solenoide finger. But it should NOT be left high long, or the solenoide can burn out
const int BUTTON_PIN = 8; // pushing button will pull pin low (pin is configured input pullup with button tied to ground on other side of button)
//const int LATENCY_TEST_PIN=6; // wire to ground through a switch to test finger latency

const int PULSE_TIME_MS = 40; // pulse time in ms to drive finger out
const int  HOLD_DUTY_CYCLE = 50; // duty cycle for PWM output to hold finger out, range 0-255 for analogWrite
const int HEARTBEAT_PERIOD_MS=500; // half cycle of built-in LED heartbeat to show we are running

const char CMD_ACTIVATE_FINGER = '1', CMD_RELAX_FINGER = '0', CMD_FLASH_LED='f'; // python sends character '1' to activate finger, '0' to relax it
// flash command is to measure end to end latency by triggering a DVS frame with LED flashing, then activate finger after inference (ignoring inference)
const char CMD_INC_PT='+', CMD_DEC_PT='-', CMD_INC_DC=']', CMD_DEC_DC='[';  // to tune values
const byte STATE_IDLE = 0, STATE_FINGER_PUSHING_OUT = 1, STATE_FINGER_HOLDING = 2;

unsigned long fingerActivatedTime = 0, heartbeatToggleTimeMs=0;
byte state = 0, previousState = state, previousButState = HIGH;
bool heartbeatFlag=0;

unsigned long pulseTimeMs; // read from EEPROM
int holdDutyCycle;

void setup()
{

  pinMode(FINGER_PIN, OUTPUT); // do this first to make sure solenoid turned off
  digitalWrite(FINGER_PIN,HIGH);  // HIGH turns OFF the finger solenoid by pulling power MOSFET gate low
  pinMode(BUTTON_PIN, INPUT_PULLUP); // button activates solenoid
  pinMode(LED_BUILTIN, OUTPUT);
//  pinMode(LATENCY_TEST_PIN, INPUT_PULLUP);

  // check EEPROM values for pulses time and duty cycle
  byte b=EEPROM.read(0);
  if(b==0) { // if eeprom holds zero we assume it has not stored a value yet, so store default value
    EEPROM.write(0, PULSE_TIME_MS); // if uninitialized, set the value
    pulseTimeMs=PULSE_TIME_MS;
  }else{
    pulseTimeMs=b; // otherwise read it
  } 
  b=EEPROM.read(1);
  if(b==0) {
    EEPROM.write(1, HOLD_DUTY_CYCLE);
    holdDutyCycle=HOLD_DUTY_CYCLE;
  }else{
    holdDutyCycle=b;
  }
  
  Serial.begin(115200);  // initialize serial communications at max speed possible 115kbaud bps
  Serial.println(VERSION);
  Serial.print("Compile date and time: ");
  Serial.print(__DATE__);
  Serial.print(" ");
  Serial.println(__TIME__);
  Serial.print("Finger pulse time in ms: ");
  Serial.println(pulseTimeMs);
  Serial.print("Finger hold duty cycle of 255: ");
  Serial.println(holdDutyCycle);
  Serial.println(HELP);
  if (DEBUG) Serial.println("*** WARNING: Compiled with DEBUG=true (will be SLOW)"); else Serial.println("Compiled with DEBUG=false");
  
  
  watchdog.enable(Watchdog::TIMEOUT_1S);
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
      case CMD_INC_PT:
        if(pulseTimeMs<255) pulseTimeMs++;
        EEPROM.write(0, pulseTimeMs);
        Serial.print("pulseTimeMs: ");
        Serial.println(pulseTimeMs);
        break;
      case CMD_INC_DC:
        if(holdDutyCycle<255) holdDutyCycle++;
        EEPROM.write(1, holdDutyCycle);
        Serial.print("holdDutyCycle: ");
        Serial.println(holdDutyCycle);
        break;
     case CMD_DEC_PT:
        if(pulseTimeMs>0) pulseTimeMs--;
        EEPROM.write(0, pulseTimeMs);
        Serial.print("pulseTimeMs: ");
        Serial.println(pulseTimeMs);
        break;
      case CMD_DEC_DC:
        if(holdDutyCycle>0) holdDutyCycle--;
        EEPROM.write(1, holdDutyCycle);
        Serial.print("holdDutyCycle: ");
        Serial.println(holdDutyCycle);
        break;
      default:
        Serial.print("unknown command character recieved: ");
        Serial.println(c);
    }
  }else if(state==STATE_IDLE){ // no serial port cmd and not active, then pay attention to button

    bool but=digitalRead(BUTTON_PIN);
  
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
    previousButState=but;
  }

  switch (state) {
    case STATE_IDLE:
      if (previousState != STATE_IDLE) {
        if (DEBUG) Serial.println("relaxing finger");
      }
      digitalWrite(FINGER_PIN, 1);
      break;
    case STATE_FINGER_PUSHING_OUT:
      if (previousState == STATE_IDLE) {
        fingerActivatedTime = millis();
        if (DEBUG) Serial.println("pushing finger out");
        digitalWrite(FINGER_PIN, 0);
      } else if (millis() - fingerActivatedTime > pulseTimeMs) {
        state = STATE_FINGER_HOLDING;
        analogWrite(FINGER_PIN, 255-holdDutyCycle); // invert because pin output is active low to turn on solenoid current
        if (DEBUG) Serial.println("now holding finger out");
      }
//      if(digitalRead(LATENCY_TEST_PIN)==0){
//          long latency=millis()-fingerActivatedTime;
//          Serial.print("measured latency in ms: ");
//          Serial.println(latency);
//      }
      break;
    case STATE_FINGER_HOLDING:
      break;
  }
  previousState = state;



  if(millis()-heartbeatToggleTimeMs>HEARTBEAT_PERIOD_MS){
    digitalWrite(LED_BUILTIN, heartbeatFlag);
    heartbeatFlag=!heartbeatFlag;
    heartbeatToggleTimeMs=millis();
    watchdog.reset();
    if(DEBUG) Serial.println("heartbeat");
  }
 
}
