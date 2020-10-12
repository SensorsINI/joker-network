
// finger control for trixsie robot
// last modification tobi Oct 2020

const String VERSION = "Trixsie Oct 2020 V2";
const String HELP="Send char 1 to activate finger, 0 to withdraw it";

const long PULSE_TIME_MS=50; // pulse time in ms to drive finger out
const float HOLD_DUTY_CYCLE=0.1; // duty cycle for PWM output to hold finger out
const int fingerPin = 13;  // setting this high will activate solenoide finger. But it should NOT be left high long, or the solenoide can burn out
const int ledPin=14; // LED to show output visually

const char CMD_ACTIVATE_FINGER='1', CMD_RELAX_FINGER='0'; // python sends character '1' to activate finger, '0' to relax it

String readString;

void setup()
{

  Serial.begin(115200);  // initialize serial communications at max speed possible 115kbaud bps
  Serial.println(VERSION);
  Serial.println(HELP);

}

void loop()
{
  while (!Serial.available()) {} // wait for data to arrive. Commands are single bytes.
  // serial read section
  while (Serial.available()) // this will be skipped if no data present, leading to
                             // the code sitting in the delay function below
  {
    delay(30);  //delay to allow buffer to fill 
    if (Serial.available() >0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readString = c; //makes the string readString
    }
  }
  if (readString.length() >0)
  {
//    Serial.println(readString.length());
//    char flag = readString.substring(1, readString.length()); ;
    Serial.print("Arduino received: ");  
    Serial.println(readString); //see what was received
    int num = readString.toInt();
    Serial.print("this is num "); 
    Serial.println(num); 
    if (num == 1){
//      Serial.println("this is 1 "); 
      digitalWrite(fingerPin, LOW);
//   Serial.println("hello");
//   delay(1000);
   
      }
      else 
      {
//        Serial.println("this is 0"); 
        digitalWrite(fingerPin, HIGH);
//   Serial.println("world");
//   delay(1000);
        }

    
//    Serial.print(" flag: ");  
//    Serial.println(flag);
  }

  delay(500);

  // serial write section

  char ard_sends = '1';
  Serial.print("Arduino sends: ");
  Serial.println(ard_sends);
  Serial.print("\n");
  Serial.flush();
//  Serial.flushInput();

//  while (Serial.available()>=0){}
}
