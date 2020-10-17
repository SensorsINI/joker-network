"""
consumer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Shasha Guo, Yuhaung Hu, Min Liu, Tobi Delbruck Oct 2020
"""
import os
import pickle

import cv2
import sys
import tensorflow as tf
# from keras.models import load_model
import serial
import time
import socket
import numpy as np
from globals_and_utils import *
from engineering_notation import EngNumber  as eng # only from pip
import collections
from pathlib import Path

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

# model = load_model(MODEL)
# tflite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_LITE)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

serial_port = sys.argv[1]
log.info('opening serial port {} to send commands to finger'.format(serial_port))
arduino_serial_port = serial.Serial(serial_port, 115200, timeout=5)
udpbufsize = IMSIZE * IMSIZE+1000
saved_non_jokers= collections.deque(maxlen=NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER) # lists of images to save
Path(JOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(NON_JOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
def next_path_index(path):
    l=os.listdir(path)
    if len(l)==0:
        return 0
    else:
        l2=sorted(l)
        next=int(l2[-1][0:-4])+1 # strip .png
        return next

next_joker_index=next_path_index(JOKERS_FOLDER)
next_non_joker_index=next_path_index(NON_JOKERS_FOLDER)

log.info('GPU is {}'.format('available' if len(tf.config.list_physical_devices('GPU'))>0 else 'not available (check tensorflow/cuda setup)'))

if __name__ == '__main__':
    while True:
        with Timer('recieve UDP'):
            receive_data, client_address = server_socket.recvfrom(udpbufsize)

        with Timer('unpickle and normalize/reshape'):
            img = pickle.loads(receive_data)
            # img = (1./255)*np.reshape(img, [IMSIZE, IMSIZE,1])
            img = (1./255)*np.reshape(img, [1,IMSIZE, IMSIZE,1])
        with Timer('run CNN'):
            interpreter.set_tensor(input_details[0]['index'], np.array(img,dtype=np.float32))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            # pred = model.predict(img[None, :])
            dec = np.argmax(pred[0])
            joker_prob=pred[0][1]

        if dec==1: # joker
            arduino_serial_port.write(b'1')
        else:
            arduino_serial_port.write(b'0')

        outimg=img.squeeze()*255
        if dec==1: # joker
            n=f'{JOKERS_FOLDER}/{next_joker_index:04d}.png'
            cv2.imwrite(n, outimg)
            next_joker_index+=1
            for i in saved_non_jokers:
                n = f'{NON_JOKERS_FOLDER}/{next_non_joker_index:04d}.png'
                cv2.imwrite(n,i)
                next_non_joker_index+=1
            saved_non_jokers.clear()
        else:
            saved_non_jokers.append(outimg)

        scale=20
        njoker_star=int(joker_prob*scale)
        log.info('joker prediction: {}{} {}'.format('*'*njoker_star,' '*(scale-njoker_star),'(JOKER)' if dec==1 else '       '))

        if dec == 1:
            time.sleep(FINGER_OUT_TIME_S)


