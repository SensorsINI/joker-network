"""
consumer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Shasha Guo, Yuhaung Hu, Min Liu, Tobi Delbruck Oct 2020
"""
import os
import cv2
import sys
from keras.models import load_model
import serial
from datetime import datetime
import time
import socket
import numpy as np
from globals_and_utils import *

log=my_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("", PORT)
server_socket.bind(address)
modelpath = MODEL
resize = IMSIZE
serial_port = sys.argv[1]
arduino_serial_port = serial.Serial(serial_port, 115200, timeout=5)
color = (0, 255, 0)
model = load_model(modelpath)
pixelnum = resize * resize

if __name__ == '__main__':
    while True:
        try:
            stime = datetime.now()
            receive_data, client_address = server_socket.recvfrom(pixelnum)
            receive_data = np.fromstring(receive_data, dtype=np.uint8)
            etime = datetime.now()
            log.info('receive time {:.2f} ms'.format((etime - stime).microseconds / 1000))

            stime = datetime.now()
            img1 = np.reshape(receive_data, [resize, resize,1])
            img1 = img1/255.
            tmp = model.predict(img1[None, :])
            pred = list(tmp[0])
            index = pred.index(max(pred))
            etime = datetime.now()
            log.info('prediction index: {}, prediction time {:.2f} ms'.format(index, (etime - stime).microseconds / 1000))

            stime = datetime.now()
            if index==1: # joker
                arduino_serial_port.write(b'1')
                log.info('JOKER!!!!! delaying 1s')
                time.sleep(1)
            else:
                arduino_serial_port.write(b'0')
            etime = datetime.now()
            log.info('serial transmission time {:.2f} ms'.format((etime - stime).microseconds / 1000))

        except Exception as e:
            log.error(str(e))
