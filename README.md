# joker-network

# Requirements:
Test environment:

OS: Mac OSX 10.15.7

python: 3.7.3

tensorflow: 1.14.0

Keras: 2.3.1

pip install opencv-python tensorflow keras pyserial

Some issues occured for Mac OSX:
 a. Error [Errno 40] Message too long. This is caused by too small UDP packet size is set, default value is only 9216. 
 Use this command to change the value to 65535: sudo sysctl -w net.inet.udp.maxdgram=65535

 b. os.environ['KMP_DUPLICATE_LIB_OK']='True' might be required to enable duplicate openMPs running, Otherwise
there might be some errors about openMP reported.


# How to run it?
connect hardware

open two terminal

run consumerudp.py
python consumerudp.py cnnModelName resize arduinoPort
example: python consumerudp.py afnorm224v1.h5 224 arduinoPort

run producerudp.py
python producerudp.py resize
example: python producerudp.py 224

open ArduinoControl/timerpwm2/timerpwm2.ino

joker-classify-network.py is for training.  
command python joker-classify-network.py imgdir weightfilename resize
imgdir is where your imgs are organized with three sub-folders: train/, valid/, and test/.  
Each sub-folder contains two sub-folders, class1/ and class2/. Class1/ includes non-joker images and class2/ contains joker images.

joker-classify-accuracy.py is for getting accuracy with pre-trained model.  
