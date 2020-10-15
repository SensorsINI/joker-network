"""
consumer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Shasha Guo, Yuhaung Hu, Min Liu, Tobi Delbruck Oct 2020
"""
import os
import pickle

import cv2
import sys
import tensorflow as tf
from keras.models import load_model
import serial
from datetime import datetime
import time
import socket
import numpy as np
from globals_and_utils import *
from timers import Timer

log=my_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

log.info('opening UDP port {} to receive frames from producer'.format(PORT))
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("", PORT)
server_socket.bind(address)
log.info('loading CNN model {}'.format(MODEL))
model = load_model(MODEL,compile=True)
serial_port = sys.argv[1]
log.info('opening serial port {} to send commands to finger'.format(serial_port))
arduino_serial_port = serial.Serial(serial_port, 115200, timeout=5)

udpbufsize = IMSIZE * IMSIZE+1000

log.info('GPU is {}'.format('available' if len(tf.config.list_physical_devices('GPU'))>0 else 'not available (check tensorflow/cuda setup)'))

if __name__ == '__main__':
    while True:
        try:
            with Timer('recieve UDP'):
                receive_data, client_address = server_socket.recvfrom(udpbufsize)

            with Timer('unpickle and normalize/reshape'):
                img = pickle.loads(receive_data)
                img = (1./255)*np.reshape(img, [IMSIZE, IMSIZE,1])
            with Timer('run CNN'):
                tmp = model.predict(img[None, :])
            with Timer('process output vector'):
                pred = list(tmp[0])
                index = pred.index(max(pred))
            log.info('{}'.format('.' if index==0 else 'joker'))

            with Timer('transmit to serial port'):
                if index==1: # joker
                    arduino_serial_port.write(b'1')
                    log.info('JOKER!!!!! delaying 1s')
                    time.sleep(1)
                else:
                    arduino_serial_port.write(b'0')

        except Exception as e:
            log.error(str(e))
