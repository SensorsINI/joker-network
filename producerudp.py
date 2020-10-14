"""
producer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Yuhuang Hu, Shasha Guo,  Min Liu, Tobi Delbruck Oct 2020
"""

import atexit
from pyaer.davis import DAVIS
import cv2
import sys
import math
from time import time
import numpy.ma as ma
import socket
import numpy as np
from globals_and_utils import *

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

def producer():
    a=time()
    frame = np.zeros([])
    if SHOW_DVS_OUTPUT:
        cv2.namedWindow('DVS', cv2.WINDOW_NORMAL)
    cv2_resized=False

    while True:
        try:
            i = 0
            leng = len(frame.shape)
            data = device.get_event()
            # assemble 'frame' of EVENT_COUNT events
            if data is not None:
                (pol_events, num_pol_event,
                 special_events, num_special_event,
                 frames_ts, frames, imu_events,
                 num_imu_event) = data
                log.debug('got {} new DVS events, before had total {} events'.format(num_pol_event, 0 if leng==0 else frame.shape[0]))
                if num_pol_event > 0:
                    if leng > 0: # if we initialized frame array already
                        if frame.shape[0] > EVENT_COUNT:
                            frame = frame[:EVENT_COUNT] # if we already got too many, discard earlier events
                        else:
                            frame = np.vstack([frame, pol_events]) # otherwise tack new events to end
                            if frame.shape[0] > EVENT_COUNT:
                                frame = frame[:EVENT_COUNT] # if too many, discard earliest ones
                            else:
                                continue # if not enough events yet, get more events
                    else: # initialize frame
                        a = time() # time we start collecting new frame of EVENT_COUNT events
                        frame = pol_events
                        if frame.shape[0] > EVENT_COUNT:
                            frame = frame[:EVENT_COUNT]  # if too many, discard earlier ones
                        else:
                            continue # get more events

                    # we get here once we have assembled EVENT_COUNT events in frame
                    b = time()
                    log.info('got frame of {} events in {:.2f} ms; making normalized 2d histogram to send'.format(frame.shape[0], (b-a)*1000))
                    a=b # start timing normalization code
                    # take DVS coordinates and scale x and y to output frame dimensions using flooring math
                    xfac=float(IMSIZE)/device.dvs_size_X
                    yfac=float(IMSIZE)/device.dvs_size_Y
                    frame[:,1]=np.floor(frame[:,1]*xfac)
                    frame[:,2]=np.floor(frame[:,2]*yfac)
                    pol_on = (frame[:, 3] == 1)
                    pol_off = np.logical_not(pol_on)

                    img_on, _, _ = np.histogram2d(
                            frame[pol_on, 2], frame[pol_on, 1],
                            bins=(IMSIZE, IMSIZE), range=histrange)
                    img_off, _, _ = np.histogram2d(
                            frame[pol_off, 2], frame[pol_off, 1],
                            bins=(IMSIZE, IMSIZE), range=histrange)
                    imgtmp = img_on - img_off # diff of both 2d histograms, ON positive counts, OFF negative counts

                    tmpvar = np.reshape(imgtmp, [imgtmp.shape[0] * imgtmp.shape[1],1]) # make unit8 vector of counts
                    tmpvar = tmpvar * 1. / np.max(tmpvar)
                    SUM = np.sum(tmpvar)
                    COUNT = np.sum(tmpvar > 0)
                    MEAN = SUM / COUNT
                    tmpvar2 = ma.masked_values(tmpvar, 0)
                    # mean3 = np.mean(tmpvar2.compressed())
                    var3 = np.var(tmpvar2.compressed())
                    sig = math.sqrt(var3)
                    if sig < (0.1 / 255.0):
                        sig = 0.1 / 255.0
                    numSDevs = 3.
                    mean_png_gray = 0 if RECTIFY_POLARITIES == True else (127. / 255)
                    zeroValue = mean_png_gray

                    fullrange = numSDevs * sig if RECTIFY_POLARITIES == True else (2. * numSDevs * sig)
                    halfRange = 0 if RECTIFY_POLARITIES == True else (numSDevs * sig)
                    rangenew = 1
                    tmpvar[tmpvar > 0] = ((tmpvar[tmpvar > 0.] + halfRange)*rangenew) / fullrange
                    tmpvar[tmpvar > 1] = 1.
                    tmpvar[tmpvar == 0] = mean_png_gray
                    pixmap = tmpvar * 1.0 * RANGE_NORMALIZED_FRAME
                    pixmap = np.reshape(pixmap, imgtmp.shape).astype('uint8')
                    b = time()
                    log.info('DVS frame normalization time {:.2f} ms'.format((b - a) * 1000))
                    try:
                        a = b
                        data = pixmap.tostring()
                        client_socket.sendto(data, udp_address)
                        b = time()
                        log.info('UDP transmission time {:.2f} ms'.format( (b-a)*1000))
                        if SHOW_DVS_OUTPUT:
                            min = np.min(pixmap)
                            img = ((pixmap - min) / (np.max(pixmap) - min))
                            cv2.imshow('DVS', img)
                            if not cv2_resized:
                                cv2.resizeWindow('DVS', 800, 600)
                                cv2_resized = True
                            # wait minimally since interp takes time anyhow
                            cv2.waitKey(1)

                        a=b
                    except KeyboardInterrupt:
                        print('interrupted!')
                        break
                    frame = np.zeros([]) # empty frame out
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pass
        except KeyboardInterrupt:
            device.shutdown()
            break


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
