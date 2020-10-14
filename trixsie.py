# main trixy script.
# uses multiprocessing to launch the producer (DVS) and consumer (tensorflow) in separate processes that are connected by pipe
import multiprocessing as mp
from pyaer.davis import DAVIS
import cv2
import sys
import math
from time import time
import numpy.ma as ma
import socket
import numpy as np
import atexit

IMSIZE=224
MODEL='afnorm224v1.h5'

if __name__ == '__main__':
