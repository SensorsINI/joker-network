import time
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng # only from pip

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
