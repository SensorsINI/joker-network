# joker-network

# Requirements:
Test environment:

 - OS: Mac OSX 10.15.7, Ubuntu 18.04
 - python: 3.7.3
 - tensorflow: 1.14.0
 - Keras: 2.3.1
 - pyaer
 
pip install opencv-python tensorflow keras pyserial

or

conda install keras tensorflow opencv numpy pyserial -c conda-forge

### pyaer
You also need pyaer. Clone it, then from the clone do the installation from within your conda environment



## Some issues that occur for Mac OSX:

 a. Error [Errno 40] Message too long. This is caused by too small UDP packet size is set, default value is only 9216. 
 Use this command to change the value to 65535: sudo sysctl -w net.inet.udp.maxdgram=65535

 b. os.environ['KMP_DUPLICATE_LIB_OK']='True' might be required to enable duplicate openMPs running, Otherwise
there might be some errors about openMP reported.


# How to run it?
 1. connect hardware
 1. open two terminals
 1. run consumerudp.py

python consumerudp.py cnnModelName resize arduinoPort
example: python consumerudp.py afnorm224v1.h5 224 arduinoPort

 1. run producerudp.py

python producerudp.py resize
example: python producerudp.py 224

open ArduinoControl/trixsie-firmware/trixsie-firmware.ino

joker-classify-network.py is for training.  
command python joker-classify-network.py imgdir weightfilename resize
imgdir is where your imgs are organized with three sub-folders: train/, valid/, and test/.  
Each sub-folder contains two sub-folders, class1/ and class2/. Class1/ includes non-joker images and class2/ contains joker images.

joker-classify-accuracy.py is for getting accuracy with pre-trained model.  
