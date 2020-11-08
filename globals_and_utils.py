""" Shared stuff between producer and consumer
 Author: Tobi Delbruck
 """
import logging
import math
import os
import sys
import time
from subprocess import TimeoutExpired

import cv2
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng  # only from pip

LOGGING_LEVEL = logging.INFO
PORT = 12000  # UDP port used to send frames from producer to consumer
IMSIZE = 224  # input image size, must match model
UDP_BUFFER_SIZE = int(math.pow(2, math.ceil(math.log(IMSIZE * IMSIZE + 1000) / math.log(2))))

EVENT_COUNT_PER_FRAME = 3000  # events per frame
EVENT_COUNT_CLIP_VALUE = 3  # full count value for colleting histograms of DVS events
SHOW_DVS_OUTPUT = True # producer shows the accumulated DVS frames as aid for focus and alignment
MIN_PRODUCER_FRAME_INTERVAL_MS=5.0 # inference takes about 3ms and normalization takes 1ms, hence at least 2ms
        # limit rate that we send frames to about what the GPU can manage for inference time
        # after we collect sufficient events, we don't bother to normalize and send them unless this time has
        # passed since last frame was sent. That way, we make sure not to flood the consumer
MAX_SHOWN_DVS_FRAME_RATE_HZ=15 # limits cv2 rendering of DVS frames to reduce loop latency for the producer
FINGER_OUT_TIME_S = 2  # time to hold out finger when joker is detected

DATA_FOLDER = home = '/home/tobi/Downloads/trixsyDataset/data' #'data'  # new samples stored here
NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER = 6
JOKERS_FOLDER = DATA_FOLDER + '/jokers'
NON_JOKERS_FOLDER = DATA_FOLDER + '/nonjokers'
SERIAL_PORT = "/dev/ttyUSB0"  # port to talk to arduino finger controller

LOG_DIR='logs'
SRC_DATA_FOLDER = '/home/tobi/Downloads/trixsyDataset/source_data'
TRAIN_DATA_FOLDER='/home/tobi/Downloads/trixsyDataset/training_dataset' # the actual training data that is produced by split from dataset_utils/make_train_valid_test()

MODEL_DIR='models' # where models stored
JOKER_NET_BASE_NAME='joker_net' # base name
TFLITE_FILE_NAME=JOKER_NET_BASE_NAME+'.tflite' # tflite model is stored in same folder as full-blown TF2 model
CLASS_DICT={'nonjoker':1, 'joker':2} # class1 and class2 for classifier

import signal
def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=30):
    """ get input with timeout

    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable

    :returns: the input
    :raises: TimeoutError if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout) # produce SIGALRM in `timeout` seconds
    try:
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0) # cancel alarm

def yes_or_no(question, default='y', timeout=None):
    """ Get y/n answer with default choice and optional timeout

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n'
    :param timeout: the timeout in seconds, default is None

    :returns: True or False
    """
    if default is not None and (default!='y' and default!='n'):
        log.error(f'bad option for default: {default}')
        quit(1)
    y='Y' if default=='y' else 'y'
    n='N' if default=='n' else 'n'
    while "the answer is invalid":
        try:
            to_str='' if timeout is None else f'(Timeout {default} in {timeout}s)'
            reply = str(input_with_timeout(f'{question} {to_str} ({y}/{n}): ',timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply=''
        if len(reply)==0 or reply=='':
            return True if default=='y' else False
        elif reply[0].lower() == 'y':
            return True
        if reply[0].lower() == 'n':
            return False

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
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    ch = logging.StreamHandler()
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

    def print_timing_info(self,stream=None):
        a = np.array(times[self.timer_name])
        timing_mean = np.mean(a) # todo use built in print method for timer
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        log.info('{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(self.timer_name, len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))

def print_timing_info():
    print('== Timing statistics ==')
    for k,v in times.items():  # k is the name, v is the list of times
        a = np.array(v)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        log.info('\n{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(k, len(a),
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

def write_next_image(dir:str, idx:int, img):
    """ Saves data sample image

    :param dir: the folder
    :param idx: the current index number
    :param img: the image to save
    :returns: the next index
    """
    while True:
        n=f'{dir}/{idx:04d}.png'
        if not os.path.isfile(n):
            break
        idx+=1
    try:
        cv2.imwrite(n, img)
    except Exception as e:
        log.error(f'error saving {n}: caught {e}')
    return idx

# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
