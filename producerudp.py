"""DAVIS346 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
这个版本是把predict作为主进程，但是producer在产生了4次数据后，就get不到数据了。
Device ID: 1
Device is master.
Device Serial Number: 00000002
Device String: DAVIS ID-1 SN-00000002 [4:12]
Device USB bus Number: 4
Device USB device address: 12
Device size X: 346
Device size Y: 260
Logic Version: 18
Background Activity Filter: True
Color Filter 0 <class 'int'>
False
frame shape 0
frame shape 0
get_event
data not none
produce time 11.95 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 68.867 ms
write ard time 0.107 ms
predition done
produce time 11.367 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.377 ms
write ard time 0.041 ms
predition done
produce time 7.382 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.385 ms
write ard time 0.039 ms
predition done
produce time 5.541 ms
classify recv msg (224, 224, 1)
index 0 predict time 4.105 ms
write ard time 0.046 ms
predition done
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
"""
from __future__ import print_function
from pyaer.davis import DAVIS
import cv2
import sys
import math
from time import time
import numpy.ma as ma
import socket
import numpy as np

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('', 12000)
class_name = ['joker', 'other']

#人脸识别分类器本地存储路径
cascade_path = "./jokeravi.xml"   

# modelpath = './184binv32.h5'
# modelpath = './alexnet184grayv12.h5'
# modelpath = sys.argv[1]
resize = int(sys.argv[1])
device = DAVIS(noise_filter=True)
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

clip_value = 3
histrange = [(0, v) for v in (resize, resize)] # allocate DVS frame histogram to desired output size

def get_event(device):
    data = device.get_event()
    return data

num_packet_before_disable = 0
rectifyPolarities = True
rangeNormalizeFrame = 255
CONSTNUM = 3000

img2 = np.zeros([])

def producer():

    frame = np.zeros([])
    print('frame shape', len(frame.shape))
    while True:
        try:
            i = 0
            leng = len(frame.shape)
            a = time()
            data = get_event(device)
            print('get_event')
            if data is not None:
                (pol_events, num_pol_event,
                 special_events, num_special_event,
                 frames_ts, frames, imu_events,
                 num_imu_event) = data
                if num_pol_event != 0:
                    if leng > 0:
                        if frame.shape[0] > CONSTNUM:
                            frame = frame[:CONSTNUM]
                        else:
                            frame = np.vstack([frame, pol_events])
                            if frame.shape[0] > CONSTNUM:
                                frame = frame[:CONSTNUM]
                            else:
                                continue
                    else:
                        frame = pol_events
                        if frame.shape[0] > CONSTNUM:
                            frame = frame[:CONSTNUM]
                        else:
                            continue
                    # take DVS coordinates and scale x and y to output frame dimensions using flooring math
                    xfac=float(resize)/device.dvs_size_X
                    yfac=float(resize)/device.dvs_size_Y
                    frame[:,1]=np.floor(frame[:,1]*xfac)
                    frame[:,2]=np.floor(frame[:,2]*yfac)
                    pol_on = (frame[:, 3] == 1)
                    pol_off = np.logical_not(pol_on)

                    img_on, _, _ = np.histogram2d(
                            frame[pol_on, 2], frame[pol_on, 1],
                            bins=(resize, resize), range=histrange)
                    img_off, _, _ = np.histogram2d(
                            frame[pol_off, 2], frame[pol_off, 1],
                            bins=(resize, resize), range=histrange)
                    imgtmp = img_on - img_off
                    tmpvar = np.reshape(imgtmp, [imgtmp.shape[0] * imgtmp.shape[1],1]).astype('uint8')
                    pixmap = np.zeros(tmpvar.shape)
                    n = len(pixmap)
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
                    pixmap = np.zeros(tmpvar.shape)
                    numSDevs = 3.
                    mean_png_gray = 0 if rectifyPolarities == True else (127. / 255)
                    zeroValue = mean_png_gray
                    fullscale = 1. - zeroValue
                    
                    fullrange = numSDevs * sig if rectifyPolarities == True else (2. * numSDevs * sig)
                    halfRange = 0. if rectifyPolarities == True else (numSDevs * sig)
                    rangenew = 1.
                    nonZeroCount = 0
                    n = len(pixmap)
                    nonZeroCount = np.sum(tmpvar > 0)
                    tmpvar[tmpvar > 0.] = ((tmpvar[tmpvar > 0.] + halfRange)*rangenew) / fullrange
                    tmpvar[tmpvar > 1.] = 1.
                    tmpvar[tmpvar == 0.] = mean_png_gray
                    pixmap = tmpvar * 1.0 * rangeNormalizeFrame
                    # print('tmpvar unique after', np.unique(pixmap))
                    fimg = np.reshape(pixmap, imgtmp.shape)
                    img1 = fimg.astype('uint8')
                    # print(' unique img2', np.unique(img2))
                    # img2 = img1 * 1. / 255

                    # classify
                    b = time()
                    # img4 = cv2.resize(img1, (resize, resize),interpolation=cv2.INTER_NEAREST)
                    # imgf = np.reshape(imgf, (resize, resize, 1))
                    # img4 = np.reshape(imgf, (resize, resize, 1))
                    print('DVS events aquisition time {:.2} ms'.format( (b-a)*1000))
                    try:
                        a = b
                        data = img1.tostring()
                        client_socket.sendto(data, server_address)
                        b = time()
                        print('communication time {:.2} ms'.format( (b-a)*1000))
                    except KeyboardInterrupt:
                        print('interrupted!')
                        break
                    frame = np.zeros([]) # todo check allocation
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
        print('Error', str(e))
        device.shutdown()
        sys.exit()
    else:
        pass
    finally:
        pass