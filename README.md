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
python -m consumer  arduinoPort
example: python -m consumer.py /dev/ttyUSB0
```

 1. run producerudp.py

```shell script
python -m producer
```

# Firmware

ArduinoControl/trixsie-firmware/trixsie-firmware.ino

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
