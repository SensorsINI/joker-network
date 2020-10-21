# joker-network

Host side code and firmware for the Trixsy card finding magic robot.

[![Watch the video](https://img.youtube.com/vi/Y0Crn4DU17M/hqdefault.jpg)](https://youtu.be/Y0Crn4DU17M)

Trixsy thrusts out its finger at the exactly correct moment as the cards are riffled past its finger.

Trixsy uses a 346x260 DVS event camera designed by the Sensors group (and sold by inivation.com) to generate constant count frames at hundreds of Hz and with a latency of  less than 1 ms. Trixy is written in python. Trisy uses https://github.com/duguyue100/pyaer to capture events in a producer process, and sends the frames to the consumer process by local UDB socket. A TensorflowLite 2.0 AlexNet CNN detects the selected card in about 4 ms. The PC sends the command to the Arduino Nano, which turns off an NPN bipolar which lets the power MOSFET gate go up to 24V. The MOSFET lets the charge stored on an 11F ultra cap array charged to 24V dump onto a solenoid to push our the finger lever in about 20 ms. Trixsy's whole finger is powered by USB bus power (24V comes from 5V VBUS via $2 DC-DC converter.)

Trixsy was developed by visiting CSC student Shasha Guo, PhD students Min Liu and Yuhuang Hu, and Tobi Delbruck. Thanks for filming to Joao Sacramento and Johannes Oswald.

See https://sensors.ini.uzh.ch for latest news and more demos

# Requirements:
Test environment:

 - OS: Fully tested on Ubuntu 18.04, partially on Mac OSX 10.15.7 
 - python: 3.8
 - tensorflow: 2.3.1
 - Keras: 2.3.1
 - pyaer https://github.com/duguyue100/pyaer
 
 **Make a conda environment**, activate it, then in it install the libraries.
 
```
pip install opencv-python tensorflow keras pyserial pyaer engineering_notation
```
or
```
conda install keras tensorflow opencv numpy pyserial -c conda-forge
pip install pyaer engineering_notation
```

### pyaer
pyaer needs https://github.com/inivation/libcaer. Clone it, then follow instructions in its README to install libcaer. 



## Some issues that occur for Mac OSX:

 a. Error [Errno 40] Message too long. This is caused by too small UDP packet size is set, default value is only 9216. 
 Use this command to change the value to 65535: sudo sysctl -w net.inet.udp.maxdgram=65535

 b. os.environ['KMP_DUPLICATE_LIB_OK']='True' might be required to enable duplicate openMPs running, Otherwise
there might be some errors about openMP reported.


# How to run it?
 1. connect hardware: DVS to usb and Arduino to USB.
 1. Find out which serial port device the Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 1. open two terminals
 1. run consumer.py

```shell script
python -m consumer  arduinoPort
example: python -m consumer.py /dev/ttyUSB0
```

 1. run producerudp.py

```shell script
python -m producer
```

# Firmware

ArduinoControl/trixsie-firmware/trixsie-firmware.ino

Output on startup on serial port is
```
*** Trixsie Oct 2020 V1.0
Compile date and time: Oct 18 2020 17*** Trixsie Oct 2020 V1.0
Compile date and time: Oct 18 2020 17:51:49
Compiled with DEBUG=false
Finger pulse time in ms: 150
Finger hold duty cycle of 255: 30
Send char '1' to activate solenoid finger, '0' to relax it
+/- increase/decrease pulse time, ]/[ increase/decrease hold duty cycle

```

# Training

joker-classify-network.py is for training.  

```
python joker-classify-network.py imgdir weightfilename resize
```

imgdir is where your imgs are organized with three sub-folders: train/, valid/, and test/.  
Each sub-folder contains two sub-folders, class1/ and class2/. Class1/ includes non-joker images and class2/ contains joker images.

joker-classify-accuracy.py is for getting accuracy with pre-trained model.  


# Results

## Runtime latencies

### consumer (with tflite model)

```
overall consumer loop n=16266: 19.07ms +/- 104.09ms (median 10.45ms, min 3.81ms max 3.25s)
recieve UDP n=16266: 12.59ms +/- 104.03ms (median 4.15ms, min 16.93us max 3.24s)
unpickle and normalize/reshape n=16265: 160.13us +/- 43.65us (median 150.44us, min 87.74us max 672.34us)
run CNN n=16265: 4.21ms +/- 901.34us (median 4.05ms, min 2.61ms max 11.24ms)
```

###producer

```
== Timing statistics ==
== Timing statistics ==
accumulate DVS n=16268: 12.32ms +/- 103.92ms (median 3.75ms, min 211.24us max 3.24s)
normalization n=16267: 2.01ms +/- 743.03us (median 1.84ms, min 854.73us max 8.45ms)
send frame n=16267: 108.55us +/- 27.40us (median 103us, min 65.57us max 475.41us)
show DVS image n=16267: 4.28ms +/- 649.69us (median 4.15ms, min 3ms max 12.77ms)
```
