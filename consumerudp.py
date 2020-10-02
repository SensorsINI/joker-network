"""DAVIS346 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
这个版本是把predict作为主进程，但是producer在产生了4次数据后，就get不到数据了。
Device ID: 1
Device is master.
Device Serial Number: 00000002
Device String: DAVIS ID-1 SN-00000002 [4:12]
Device USB bus Number: 4
Device USB device address: 12
Device size X: 346
Device size Y: 260
Logic Version: 18
Background Activity Filter: True
Color Filter 0 <class 'int'>
False
frame shape 0
frame shape 0
get_event
data not none
produce time 11.95 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 68.867 ms
write ard time 0.107 ms
predition done
produce time 11.367 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.377 ms
write ard time 0.041 ms
predition done
produce time 7.382 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.385 ms
write ard time 0.039 ms
predition done
produce time 5.541 ms
classify recv msg (224, 224, 1)
index 0 predict time 4.105 ms
write ard time 0.046 ms
predition done
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
"""
from __future__ import print_function
import cv2
import sys
from keras.models import load_model
import serial
from datetime import datetime
import time
import socket
import numpy as np
import logging
log = logging.getLogger(__name__)


MAXB = 10000000
PORT = 12000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("", PORT)
server_socket.bind(address)
class_name = ['joker', 'other']
#人脸识别分类器本地存储路径
cascade_path = "./jokeravi.xml"   
# modelpath = './184binv32.h5'
# modelpath = './alexnet184grayv12.h5'
modelpath = sys.argv[1]
resize = int(sys.argv[2])
port = sys.argv[3]
ard = serial.Serial(port,9600,timeout=5)

    
# files = os.listdir(savepath)
# savedcounter = len(files)
# print(avipath, modelpath, savedcounter)      
color = (0, 255, 0)
model = load_model(modelpath)
x = 112
y = 112
w = 50
h = 50
#循环检测识别人脸
count = 0
# savedcounter = 30000
clip_value = 3
histrange = [(0, v) for v in (260, 346)]
shapelike = [260,346,1]
def get_event(device):
    data = device.get_event()

    return data


num_packet_before_disable = 0


frame = np.zeros([])
print('frame shape', len(frame.shape))

img2 = np.zeros([])

pixelnum = resize * resize
print(pixelnum)

if __name__ == '__main__':
    while True:

        try:
            
            stime = datetime.now()

            receive_data, client_address = server_socket.recvfrom(pixelnum)
            # print(len(receive_data), type(receive_data))
            etime = datetime.now()
            # print('load image', (etime - stime).microseconds / 1000, 'ms')
            # receive_data = receive_data.decode("cp1252")
            # print(len(receive_data))
            stime = datetime.now()
            receive_data = np.fromstring(receive_data, dtype=np.uint8)
            # print(receive_data)
            # print("接收到了客户端 %s 传来的数据: %s" % (client_address, np.reshape(receive_data,[3,3,1])))
            img1 = np.reshape(receive_data, [resize, resize])
            # print('recv from udp', img1.shape)
            # img1 = cv2.imread(savepath + 'newest.jpg', 0)

            img4 = np.reshape(img1, (resize, resize, 1))

            img4 = img4 * 1. / 255
            

            etime = datetime.now()


            stime = datetime.now()

            tmp = model.predict(img4[None, :]) 

            etime = datetime.now()
            # print(pred)
            pred = list(tmp[0])
                
            index = pred.index(max(pred))
                
            # print('index', index, 'predict time', (etime - stime).microseconds / 1000, 'ms')
            # label = class_name[pred.index(max(pred))]

            stime = datetime.now()
            setTemp1 = str(index)
            ard.write(setTemp1.encode())
            etime = datetime.now()
            
            # print('write ard time', (etime - stime).microseconds / 1000, 'ms')

            # time.sleep(0.005)

            cv2.imshow("image", img4)
            if index != 0:
                print('finger index', index)

                time.sleep(0.5)



                # stime = datetime.now()
                # jpgname = savepath + str(savedcounter) + '.jpg'
                # cv2.imwrite(jpgname, img1)

                # savedcounter += 1

                # etime = datetime.now()
                # print('storage time', (etime - stime).microseconds / 1000, 'ms')

            # print('prediction done')

        except Exception as e:
            print(str(e))
