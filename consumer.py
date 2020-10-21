"""
consumer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Shasha Guo, Yuhaung Hu, Min Liu, Tobi Delbruck Oct 2020
"""
import pickle
import cv2
import sys
import tensorflow as tf
# from keras.models import load_model
import serial
import socket
from select import select
from globals_and_utils import *
from engineering_notation import EngNumber  as eng # only from pip
import collections
from pathlib import Path
import random

log=my_logger(__name__)

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

log.info('opening UDP port {} to receive frames from producer'.format(PORT))
server_socket:socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

address = ("", PORT)
server_socket.bind(address)

log.info('loading CNN model {}'.format(MODEL_LITE))

# model = load_model(MODEL)
# tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert
interpreter = tf.lite.Interpreter(model_path=MODEL_LITE)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(sys.argv)>2:
    log.error('too many arguments\nUsage producer.py [serial_port]')
elif len(sys.argv)==2:
    serial_port = sys.argv[1]
else:
    serial_port=SERIAL_PORT


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
cv2_resized=dict()
finger_out_time=0
STATE_IDLE=0
STATE_FINGER_OUT=1
state=STATE_IDLE

log.info('GPU is {}'.format('available' if len(tf.config.list_physical_devices('GPU'))>0 else 'not available (check tensorflow/cuda setup)'))

def show_frame(frame,name, resized_dict):
    """ Show the frame in named cv2 window and handle resizing
    :param frame: 2d array of float
    :param name: string name for window
    """
    cv2.namedWindow(name,  cv2.WINDOW_NORMAL)
    cv2.imshow(name, frame)
    if not (name in resized_dict):
        cv2.resizeWindow(name, 300, 300)
        resized_dict[name] = True
        # wait minimally since interp takes time anyhow
        cv2.waitKey(1)

if __name__ == '__main__':
    while True:
        timestr = time.strftime("%Y%m%d-%H%M")
        with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
            with Timer('recieve UDP'):
                # inputready,_,_ = select([server_socket],[],[],.1)
                # if len(inputready)==0:
                #     cv2.waitKey(1)
                #     continue
                receive_data, client_address = server_socket.recvfrom(udpbufsize)

            with Timer('unpickle and normalize/reshape'):
                img = pickle.loads(receive_data)
                # img = (1./255)*np.reshape(img, [IMSIZE, IMSIZE,1])
                img = (1./255)*np.reshape(img, [1,IMSIZE, IMSIZE,1])
            with Timer('run CNN'):
                # pred = model.predict(img[None, :])
                interpreter.set_tensor(input_details[0]['index'], np.array(img,dtype=np.float32))
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                dec = np.argmax(pred[0])
                joker_prob=pred[0][1]

            if dec==1: # joker
                arduino_serial_port.write(b'1')
                finger_out_time=time.time()
                state=STATE_FINGER_OUT
                log.info('sent finger OUT')
            elif state==STATE_FINGER_OUT and time.time()-finger_out_time>FINGER_OUT_TIME_S:
                arduino_serial_port.write(b'0')
                state=STATE_IDLE
                log.info('sent finger IN')
            else:
                pass

            save_img= (255 *img.squeeze()).astype('uint8')
            if dec==1: # joker
                n=f'{JOKERS_FOLDER}/{next_joker_index:04d}.png'
                cv2.imwrite(n, save_img)
                next_joker_index+=1
                show_frame(save_img, 'joker', cv2_resized)
                non_joker_window_number=0
                for i in saved_non_jokers:
                    n = f'{NON_JOKERS_FOLDER}/{next_non_joker_index:04d}.png'
                    cv2.imwrite(n,i)
                    next_non_joker_index+=1
                    show_frame(i,f'nonjoker{non_joker_window_number}', cv2_resized)
                    non_joker_window_number+=1
                saved_non_jokers.clear()
            else:
                if random.random()<.03: # append random previous images to not just get previous almost jokers
                    saved_non_jokers.append(save_img)

            # if True:
            #     scale=20
            #     njoker_star=int(joker_prob*scale)
            #     log.info('joker prediction: {}{} {}'.format('*'*njoker_star,' '*(scale-njoker_star),'(JOKER)' if dec==1 else '        '))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


