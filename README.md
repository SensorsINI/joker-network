# joker-network

Host side code for the Trixsy card magic robot.

# Requirements:
Test environment:

 - OS: Mac OSX 10.15.7 or Ubuntu 18.04
 - python: 3.8
 - tensorflow: 2.3.1
 - Keras: 2.3.1
 - pyaer https://github.com/duguyue100/pyaer
 
 Make a conda environment, then in it install the libraries.
 
```
pip install opencv-python tensorflow keras pyserial pyaer engineering_notation
```
or
```
conda install keras tensorflow opencv numpy pyserial -c conda-forge
pip install pyaer engineering_notation
```

### pyaer
pyaer needs libcaer. Clone it, then follow instructions in its README to install libcaer. 



## Some issues that occur for Mac OSX:

 a. Error [Errno 40] Message too long. This is caused by too small UDP packet size is set, default value is only 9216. 
 Use this command to change the value to 65535: sudo sysctl -w net.inet.udp.maxdgram=65535

 b. os.environ['KMP_DUPLICATE_LIB_OK']='True' might be required to enable duplicate openMPs running, Otherwise
there might be some errors about openMP reported.


# How to run it?
 1. connect hardware: DVS to usb and Arduino to USB.
 1. Find out which serial port device the Arduino appears on. You can use dmesg on linux.
 1. open two terminals
 1. run consumerudp.py

```shell script
python -m consumerudp  arduinoPort
example: python -m consumerudp.py /dev/ttyUSB0
```

 1. run producerudp.py

```shell script
python -m producerudp
```

## Firmware

ArduinoControl/trixsie-firmware/trixsie-firmware.ino

joker-classify-network.py is for training.  
command python joker-classify-network.py imgdir weightfilename resize
imgdir is where your imgs are organized with three sub-folders: train/, valid/, and test/.  
Each sub-folder contains two sub-folders, class1/ and class2/. Class1/ includes non-joker images and class2/ contains joker images.

joker-classify-accuracy.py is for getting accuracy with pre-trained model.  
