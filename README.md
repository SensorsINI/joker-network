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
pip install opencv-python tensorflow keras pyserial pyaer engineering_notation matplotlib sklearn flopco-keras psutil
```
or
```
conda install keras tensorflow opencv numpy pyserial  -c conda-forge
pip install pyaer engineering_notation sklearn  flopco-keras
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
example: python -m consumer.py 
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

_train.py_ trains the network  

```
python train.py
```

TRAIN_DATA_FOLDER is where your examples are organized with three sub-folders: _train_/, _valid_/, and _test_/.

The script _make_train_valid_test()_ in _dataset_utils.py_ builds these folders using desired split from source folders _class1_ and _class2_.  

Each train/valid/test sub-folder contains two sub-folders, _class1_/ and _class2_/. 

_class1_/ has non-joker images and _class2_/ contains joker images.


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

## Training outcome

See [training.log](training.log).

## Runtime latencies

### consumer (with tflite model)

```
== Timing statistics ==
overall consumer loop n=20772: 11.14ms +/- 35.28ms (median 6.50ms, min 2.57ms max 2.11s)
recieve UDP n=20772: 4.79ms +/- 35.09ms (median 25.03us, min 14.31us max 2.11s)
unpickle and normalize/reshape n=20771: 141.34us +/- 38.75us (median 131.61us, min 82.02us max 625.37us)
run CNN n=20771: 4.04ms +/- 894.48us (median 3.91ms, min 2.37ms max 10ms)
```

###producer

```
== Timing statistics ==
overall producer frame rate n=32463: 10.62ms +/- 66.40ms (median 5.43ms, min 1.21ms max 3.98s)
2020-10-23 12:02:37,130 - __main__ - INFO - closing <pyaer.davis.DAVIS object at 0x7fe47768e370> (producer.py:28)
2020-10-23 12:02:37,136 - globals_and_utils - INFO - saving timing data for overall producer frame rate in numpy file data/producer-frame-rate-20201023-1156.npy (globals_and_utils.py:101)
2020-10-23 12:02:37,136 - globals_and_utils - INFO - there are 32463 times (globals_and_utils.py:102)
accumulate DVS n=32463: 8.46ms +/- 66.02ms (median 3.73ms, min 154.50us max 3.97s)
normalization n=32462: 1.33ms +/- 329.35us (median 1.27ms, min 754.12us max 4.11ms)
send frame n=32462: 104.71us +/- 29.46us (median 100.61us, min 41.01us max 397.44us)
show DVS image n=2543: 4.58ms +/- 1.66ms (median 4.37ms, min 2.45ms max 64.55ms)
```
