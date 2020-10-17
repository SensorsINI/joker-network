""" Shared stuff between producer and consumer """
import logging
import os
import time
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng # only from pip


LOGGING_LEVEL = logging.INFO
MAXB = 10000000
PORT = 12000
IMSIZE=224
MODEL='afnorm224v1.h5'
MODEL_LITE='joker.tflite'
EVENT_COUNT_PER_FRAME = 3000
EVENT_COUNT_CLIP_VALUE = 3
SHOW_DVS_OUTPUT=True
FINGER_OUT_TIME_S=2
NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER=10
DATA_FOLDER='data'
JOKERS_FOLDER=DATA_FOLDER+'/jokers'
NON_JOKERS_FOLDER= DATA_FOLDER + '/nonjokers'

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def my_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger

cuda_timers = {}
timers = {}

class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        timers[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        a=np.array(timing_values)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median= np.median(a)
        timing_min=np.min(a)
        timing_max=np.max(a)
        print('{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(timer_name, len(timing_values), eng(timing_mean),eng(timing_std),eng(timing_median), eng(timing_min), eng(timing_max) ))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)

