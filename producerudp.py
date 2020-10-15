"""
producer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Yuhuang Hu, Shasha Guo,  Min Liu, Tobi Delbruck Oct 2020
"""

import atexit
import pickle

from pyaer.davis import DAVIS
import cv2
import sys
import math
from time import time
import numpy.ma as ma
import socket
import numpy as np
from globals_and_utils import *
from timers import Timer

log=my_logger(__name__)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_address = ('', PORT)

device = DAVIS(noise_filter=True)
def cleanup():
    log.info('closing {}'.format(device))
    device.shutdown()
    cv2.destroyAllWindows()

atexit.register(cleanup)

print("Device ID:", device.device_id)
if device.device_is_master:
    print("Device is master.")
else:
    print("Device is slave.")
print("Device Serial Number:", device.device_serial_number)
print("Device String:", device.device_string)
print("Device USB bus Number:", device.device_usb_bus_number)
print("Device USB device address:", device.device_usb_device_address)
print("Device size X:", device.dvs_size_X)
print("Device size Y:", device.dvs_size_Y)
print("Logic Version:", device.logic_version)
print("Background Activity Filter:",
      device.dvs_has_background_activity_filter)
print("Color Filter", device.aps_color_filter, type(device.aps_color_filter))
print(device.aps_color_filter == 1)

device.start_data_stream()
# setting bias after data stream started
device.set_bias_from_json("./configs/davis346_config.json")
xfac=float(IMSIZE)/device.dvs_size_X
yfac=float(IMSIZE)/device.dvs_size_Y

def producer():
    if SHOW_DVS_OUTPUT:
        cv2.namedWindow('DVS', cv2.WINDOW_NORMAL)
    cv2_resized=False
    try:
        while True:
            events = None
            with Timer('accumulate DVS'):
                while events is None or len(events)<EVENT_COUNT:
                    data = device.get_event()
                    # assemble 'frame' of EVENT_COUNT events
                    if data is not None:
                        (pol_events, num_pol_event,
                         special_events, num_special_event,
                         frames_ts, frames, imu_events,
                         num_imu_event) = data
                        if num_pol_event > 0:
                            if events is None:
                                events=pol_events
                            else:
                                events = np.vstack([events, pol_events]) # otherwise tack new events to end
                    log.debug('got {} events (total so far {}/{} events)'
                             .format(num_pol_event, 0 if events is None else len(events), EVENT_COUNT))

            with Timer('normalization'):
                log.info('got {} events, making frame'.format(len(events)))
                # take DVS coordinates and scale x and y to output frame dimensions using flooring math
                events[:,1]=np.floor(events[:,1]*xfac)
                events[:,2]=np.floor(events[:,2]*yfac)
                pol_on = (events[:, 3] == 1)
                pol_off = np.logical_not(pol_on)

                img_on, _, _ = np.histogram2d(
                    events[pol_on, 2], events[pol_on, 1],
                    bins=(IMSIZE, IMSIZE), range=histrange)
                img_off, _, _ = np.histogram2d(
                    events[pol_off, 2], events[pol_off, 1],
                    bins=(IMSIZE, IMSIZE), range=histrange)
                pixmap = img_on + img_off # diff of both 2d histograms, ON positive counts, OFF negative counts
                pixmap[pixmap>CLIP_VALUE]=CLIP_VALUE
                pixmap=(1./CLIP_VALUE)*pixmap
                pixmap=pixmap.astype('uint8')

                # tmpvar = np.reshape(imgtmp, [imgtmp.shape[0] * imgtmp.shape[1],1]) # make unit8 vector of counts
                # tmpvar = tmpvar * 1. / np.max(tmpvar)
                # SUM = np.sum(tmpvar)
                # COUNT = np.sum(tmpvar > 0)
                # MEAN = SUM / COUNT
                # tmpvar2 = ma.masked_values(tmpvar, 0)
                # # mean3 = np.mean(tmpvar2.compressed())
                # var3 = np.var(tmpvar2.compressed())
                # sig = math.sqrt(var3)
                # if sig < (0.1 / 255.0):
                #     sig = 0.1 / 255.0
                # numSDevs = 3.
                # mean_png_gray = 0 if RECTIFY_POLARITIES == True else (127. / 255)
                # zeroValue = mean_png_gray
                #
                # fullrange = numSDevs * sig if RECTIFY_POLARITIES == True else (2. * numSDevs * sig)
                # halfRange = 0 if RECTIFY_POLARITIES == True else (numSDevs * sig)
                # rangenew = 1
                # tmpvar[tmpvar > 0] = ((tmpvar[tmpvar > 0.] + halfRange)*rangenew) / fullrange
                # tmpvar[tmpvar > 1] = 1.
                # tmpvar[tmpvar == 0] = mean_png_gray
                # pixmap = tmpvar * 1.0 * RANGE_NORMALIZED_FRAME
                # pixmap = np.reshape(pixmap, imgtmp.shape).astype('uint8')

            with Timer('send frame'):
                data = pickle.dumps(pixmap)
                client_socket.sendto(data, udp_address)

            if SHOW_DVS_OUTPUT:
                with Timer('show DVS image'):
                    min = np.min(pixmap)
                    img = ((pixmap - min) / (np.max(pixmap) - min))
                    cv2.imshow('DVS', img)
                    if not cv2_resized:
                        cv2.resizeWindow('DVS', 800, 600)
                        cv2_resized = True
                        # wait minimally since interp takes time anyhow
                        cv2.waitKey(1)
            events = None # empty frame out
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        device.shutdown()



if __name__ == '__main__':
    try:
        producer()
    except Exception as e:
        log.error('Error', str(e))
        cleanup()
        sys.exit()
    else:
        pass
    finally:
        pass
