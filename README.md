# joker-network

Host side code and firmware for the Trixsy card finding magic robot.

[![Watch the video](https://img.youtube.com/vi/Y0Crn4DU17M/hqdefault.jpg)](https://youtu.be/Y0Crn4DU17M)

Trixsy thrusts out its finger at the exactly correct moment as the cards are riffled past its finger.

Trixsy uses a 346x260 DVS event camera designed by the Sensors group (and sold by inivation.com) to generate constant count frames at hundreds of Hz and with a latency of  less than 1 ms. Trixy is written in python. Trisy uses https://github.com/duguyue100/pyaer to capture events in a producer process, and sends the frames to the consumer process by local UDB socket. A TensorflowLite 2.0 AlexNet CNN detects the selected card in about 4 ms. The PC sends the command to the Arduino Nano, which turns off an NPN bipolar which lets the power MOSFET gate go up to 24V. The MOSFET lets the charge stored on an 11F ultra cap array charged to 24V dump onto a solenoid to push our the finger lever in about 20 ms. Trixsy's whole finger is powered by USB bus power (24V comes from 5V VBUS via $2 DC-DC converter.)

Trixsy was developed by Tobi Delbruck, visiting CSC student Shasha Guo, and PhD students Min Liu and Yuhuang Hu. Thanks for filming to Joao Sacramento and Johannes Oswald.

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
pip install opencv-python tensorflow keras pyserial pyaer engineering_notation matplotlib
```
or
```
conda install keras tensorflow opencv numpy pyserial -c conda-forge
pip install pyaer engineering_notation
```

#### Note about pip vs conda
For some reason, pip is preferred over conda for installing opencv and tensorflow. At least at time of this file.

### pyaer
pyaer needs https://github.com/inivation/libcaer. Clone it, then follow instructions in its README to install libcaer. 



## Some issues that occur for Mac OSX:

 a. Error [Errno 40] Message too long. This is caused by too small UDP packet size is set, default value is only 9216. 
 Use this command to change the value to 65535: sudo sysctl -w net.inet.udp.maxdgram=65535

 b. os.environ['KMP_DUPLICATE_LIB_OK']='True' might be required to enable duplicate openMPs running, Otherwise
there might be some errors about openMP reported.


# How to run it?
 1. connect hardware: DVS to USB and Arduino to USB.
 1. Find out which serial port device the Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 1. In first terminal run producer
```shell script
python -m producer
```
 2. In a second terminal, run consumer
```shell script
python -m consumer  arduinoPort
example: python -m consumer.py /dev/ttyUSB0
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

_joker-classify-network.py_ is for training.  

```
python joker-classify-network.py imgdir weightfilename resize
```

_imgdir_ is where your imgs are organized with three sub-folders: _train_/, _valid_/, and _test_/.  

Each sub-folder contains two sub-folders, _class1_/ and _class2_/. _class1_/ includes non-joker images and _class2_/ contains joker images.

_joker-classify-accuracy.py_ is for getting accuracy with pre-trained model.  


# Results

## Model
```
2020-10-22 22:34:29.078718: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
dataset path: TRAIN_DATA_FOLDER=/home/tobi/Downloads/trixsyDataset

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1 (Conv2D)               (None, 54, 54, 64)        7808      
_________________________________________________________________
batch_normalization (BatchNo (None, 54, 54, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 26, 26, 64)        0         
_________________________________________________________________
conv2 (Conv2D)               (None, 26, 26, 64)        102464    
_________________________________________________________________
batch_normalization_1 (Batch (None, 26, 26, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
conv3 (Conv2D)               (None, 12, 12, 128)       73856     
_________________________________________________________________
conv4 (Conv2D)               (None, 12, 12, 128)       147584    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 3200)              0         
_________________________________________________________________
fc8 (Dense)                  (None, 100)               320100    
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
output (Dense)               (None, 2)                 202       
=================================================================
Total params: 652,526
Trainable params: 652,270
Non-trainable params: 256

Found 33555 images belonging to 2 classes.
Found 3018 images belonging to 2 classes.
Found 0 images belonging to 2 classes.
 1/300 [..............................] - ETA: 0s - loss: 2.1360 - accuracy: 0.2969
263/300 [=========================>....] - 118s 448ms/step - loss: 0.4770 - accuracy: 0.8034 - val_loss: 0.5494 - val_accuracy: 0.8131
2020-10-22 22:36:28.053056 __main__ - INFO - saving model to {filename} (joker-classify-network.py:297)

```

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
overall producer frame rate n=5369: 2ms +/- 1.24ms (median 1.57ms, min 149.73us max 53.78ms)
accumulate DVS n=5369: 408.20us +/- 366.78us (median 252.01us, min 94.65us max 3.22ms)
normalization n=5368: 1.15ms +/- 392.04us (median 1.01ms, min 737.43us max 5.75ms)
send frame n=5368: 81.26us +/- 56.89us (median 56.98us, min 43.39us max 562.91us)
show DVS image n=106: 4.54ms +/- 4.78ms (median 3.52ms, min 2.66ms max 50.51ms)
```
