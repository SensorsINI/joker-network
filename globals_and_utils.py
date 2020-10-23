""" Shared stuff between producer and consumer """
import logging
import os
import time
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng  # only from pip

LOGGING_LEVEL = logging.INFO
PORT = 12000  # UDP port used to send frames from producer to consumer
IMSIZE = 224  # input image size, must match model
# MODEL='afnorm224v1.h5'
MODEL_LITE = 'joker.tflite'  # joker network model
EVENT_COUNT_PER_FRAME = 3000  # events per frame
EVENT_COUNT_CLIP_VALUE = 3  # full count value for colleting histograms of DVS events
SHOW_DVS_OUTPUT = True # producer shows the accumulated DVS frames as aid for focus and alignment
MAX_SHOWN_DVS_FRAME_RATE_HZ=10 # limits cv2 rendering of DVS frames to reduce loop latency for the producer
FINGER_OUT_TIME_S = 2  # time to hold out finger when joker is detected
DATA_FOLDER = 'data'  # new samples stored here
NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER = 6
JOKERS_FOLDER = DATA_FOLDER + '/jokers'
NON_JOKERS_FOLDER = DATA_FOLDER + '/nonjokers'
SERIAL_PORT = "/dev/ttyUSB0"  # port to talk to arduino finger controller
TRAIN_DATA_FOLDER='/home/tobi/Downloads/trixsyDataset'

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


log=my_logger(__name__)

timers = {}
times = {}
class Timer:
    def __init__(self, timer_name='', show_hist=False, numpy_file=None):
        self.timer_name = timer_name
        self.show_hist = show_hist
        self.numpy_file = numpy_file

        if self.timer_name not in timers.keys():
            timers[self.timer_name] = self
        if self.timer_name not in times.keys():
            times[self.timer_name]=[]

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        times[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')
    for k,v in times.items():  # k is the name, v is the list of times
        a = np.array(v)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        print('{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(k, len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))
        if timers[k].numpy_file is not None:
            try:
                log.info(f'saving timing data for {k} in numpy file {timers[k].numpy_file}')
                log.info('there are {} times'.format(len(a)))
                np.save(timers[k].numpy_file, a)
            except Exception as e:
                log.error(f'could not save numpy file {timers[k].numpy_file}; caught {e}')

        if timers[k].show_hist:
            from matplotlib import pyplot as plt
            dt = np.log10(np.clip(np.array(v),1e-6, None))
            plt.hist(dt,bins=100)
            plt.xlabel('log10(interval[ms])')
            plt.ylabel('frequency')
            plt.title(k)
            plt.show()


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
