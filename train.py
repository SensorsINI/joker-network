# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train/ valid/ test/ folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
# author: Tobi Delbruck
import argparse
import datetime
import glob
import shutil
from pathlib import Path
import random
from shutil import copyfile
from tkinter import *
from tkinter import filedialog, simpledialog

# uncomment lines to run on CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import PIL
import tensorflow as tf
import tensorflow.python.keras
from classification_models.keras import Classifiers
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
# from alessandro: use keras from tensorflow, not from keras directly
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from globals_and_utils import *

INITIALIZE_MODEL_FROM_LATEST = True  # set True to initialize weights to latest saved model

import logging
import sys

log = my_logger(__name__)


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def rename_images():
    """ Cleans up a folder filled with images (png and jpg) so that the images are numbered consecutively. Useful after using mv --backup=t to add new images to a folder
    :param folder: the folder name to clean up, relative to working directory
    """
    folder = SRC_DATA_FOLDER
    root = Tk()
    root.withdraw()
    os.chdir(folder)
    folder = filedialog.askdirectory()
    if len(folder) == 0:
        log.info('aborted')
        quit(1)
    os.chdir(folder)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    ls = os.listdir()  # glob.glob('./*')
    ls = sorted(ls, key=lambda f: os.stat(f).st_mtime)
    log.info(f'folder {folder} has {len(ls)} files')
    i = 0
    log.info('renaming files to tmp folder')
    for f in tqdm(ls):
        if 'png' in f:
            fn = f'tmp/{i:06d}.png'
            i = i + 1
            os.rename(f, fn)
        elif 'jpg' in f:
            fn = f'tmp/{i:06d}.jpg'
            i = i + 1
            os.rename(f, fn)

    os.chdir('tmp')
    ls = os.listdir()
    log.info('moving files back to src folder')
    for f in tqdm(ls):
        os.rename(f, '../' + f)
    os.chdir('..')
    os.rmdir('tmp')


def make_training_set():
    """ Generates the train/ valid/ and test/ folders in TRAIN_DATA_FOLDER  using the images in SRC_DATA_FOLDER and the split defined by SPLIT variable
    """
    NUM_FRAMES_PER_SEGMENT = 20  # each 'sample' comes from a consecutive sequence of this many frames to try to avoid that valid/test set have frames that are next to training set frames
    NUM_FRAMES_GAP_BETWEEN_SEGMENTS = 5  # each 'sample' comes from a consecutive sequence of this many frames to try to avoid that valid/test set have frames that are next to training set frames
    SPLIT = {'train': .8, 'valid': .1, 'test': .1}
    show_preview_frames = yes_or_no('show samples from dataset?', default='n', timeout=30)

    log.info(f'making training set from {SRC_DATA_FOLDER} with segments of {NUM_FRAMES_PER_SEGMENT} consecutive images with gaps of {NUM_FRAMES_GAP_BETWEEN_SEGMENTS} frames between segments')
    if not os.path.isdir(TRAIN_DATA_FOLDER):
        log.warning(f'{TRAIN_DATA_FOLDER} does not exist, creating it')
        Path(TRAIN_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    else:
        timestr = time.strftime("%Y%m%d-%H%M")
        backup_folder = f'{TRAIN_DATA_FOLDER}_{timestr}'
        log.warning(f'Renaming existing training folder {TRAIN_DATA_FOLDER} to {backup_folder}')
        try:
            os.rename(TRAIN_DATA_FOLDER, backup_folder)
        except OSError as e:
            log.error('target to rename existing training set folder exists already probably because you just ran this script')
            quit(1)

    log.info(f'Using source images from {SRC_DATA_FOLDER}')
    os.chdir(SRC_DATA_FOLDER)
    last_frame_time = time.time()
    SPLIT = {'train': .8, 'valid': .1, 'test': .1}
    for cls in ['class1', 'class2']:
        ls = os.listdir(cls)
        ls = sorted(ls)  # sort first to get in number order
        nfiles = len(ls) - 1
        nsegments = nfiles // (NUM_FRAMES_PER_SEGMENT + NUM_FRAMES_GAP_BETWEEN_SEGMENTS)
        segs = list(range(nsegments))
        random.shuffle(segs)  # shuffle segments of batches of images
        log.info(f'{cls} has {nfiles} samples that are split to {nsegments} sequences of {NUM_FRAMES_PER_SEGMENT} frames/seq')
        split_start = 0
        for split_name, split_frac in zip(SPLIT.keys(), SPLIT.values()):
            dest_folder_name = os.path.join(TRAIN_DATA_FOLDER, split_name, cls)
            log.info(f'making {dest_folder_name}/ folder for shuffled segments of {NUM_FRAMES_PER_SEGMENT} frames per segment for {split_frac * 100:.1f}% {split_name} split of {cls} ')
            Path(dest_folder_name).mkdir(parents=True, exist_ok=True)
            split_end = split_start + split_frac
            seg_range = range(math.floor(split_start * nsegments), math.floor(split_end * nsegments))
            file_nums = []
            print('taking segments ', end='')
            nperline = 20
            line = 0
            for s in seg_range:
                print(f'{segs[s]} ', end='')
                line += 1
                if line % nperline == 0:
                    print('')
                start = segs[s] * (NUM_FRAMES_PER_SEGMENT + NUM_FRAMES_GAP_BETWEEN_SEGMENTS)
                end = start + NUM_FRAMES_PER_SEGMENT
                nums = list(range(start, end))
                file_nums.extend(nums)
            print('')
            files = [ls[i] for i in file_nums]
            for file_name in tqdm(files, desc=f'{cls}/{split_name}'):
                source_file_path = os.path.join(SRC_DATA_FOLDER, cls, file_name)
                if file_name.lower().endswith('jpg'):
                    base = os.path.splitext(file_name)[0]
                    file_name = base + '.png'
                dest_file_path = os.path.join(dest_folder_name, file_name)
                try:
                    img = tf.keras.preprocessing.image.load_img(source_file_path, color_mode='grayscale')
                except PIL.UnidentifiedImageError:
                    log.warning(f'{source_file_path} is not an image')
                    continue
                if img.size == (IMSIZE, IMSIZE) and img.format == 'PNG':
                    # print(f'copying {source_file_path} -> {dest_file_path}')
                    copyfile(source_file_path, dest_file_path)
                    img_arr = None
                else:
                    # print(f'resizing and converting{source_file_path} -> {dest_file_path}')
                    img_arr = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
                    img_arr = tf.image.resize(img_arr, (IMSIZE, IMSIZE))  # note we do NOT want black borders and to preserve aspect ratio! We just want to squash to square
                    img_arr = np.array(img_arr, dtype=np.uint8)
                    try:
                        tf.keras.preprocessing.image.save_img(dest_file_path, img_arr, file_format='png', scale=True)
                    except Exception as e:
                        log.warning(f'{dest_file_path} could not be saved: {e}')
                        continue
                if show_preview_frames and time.time() - last_frame_time > 1:
                    cv2.namedWindow(cls, cv2.WINDOW_NORMAL)
                    cv2.imshow(cls, img_arr if img_arr is not None else tf.keras.preprocessing.image.img_to_array(img, dtype='uint8'))
                    cv2.waitKey(1)
                    last_frame_time = time.time()

            split_start = split_end
    log.info(f'done generating training set from')


def riffle_test(args):
    """ Runs test on folder of video sequence and pause at detected jokers
    """

    class GetOutOfLoop(Exception):
        pass

    start_timestr = time.strftime("%Y%m%d-%H%M")
    log.info('evaluating riffle')
    log.info(f'Tensorflow version {tf.version.VERSION}')

    def print_help():
        print('select folder (go inside it to select it)\n'
              'q|x exits\n'
              'space plays\n'
              '. forward\n'
              ', backwards\n'
              'j moves to joker folder\n'
              'n moves to nonjoker folder\n'
              'g goes to selected frame (by dialog for frame number)\n'
              'f/r fastfowards/rewinds backwards\n'
              't toggles between pausing at possible joker|pausing only at certain jokers|not pausing\n'
              'enter selects new playback folder\n'
              'h print help')

    pause_modes={0:'pause_possible',1:'pause_certain',2:'dont_pause'}
    pause_mode=0

    interpreter, input_details, output_details = load_tflite_model()
    Path(JOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(NONJOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    jokers_list_file = open(os.path.join(LOG_DIR, f'joker-file-list-{start_timestr}.txt'), 'w')
    nonjokers_list_file = open(os.path.join(LOG_DIR, f'nonjoker-file-list-{start_timestr}.txt'), 'w')

    def move_with_backup(src,dest):
        if not os.path.isdir(dest):
            raise Exception(f'destination {dest} should be a directory')
        try:
            shutil.move(src, dest)
        except Exception as e:
            idx=1
            s=os.path.splitext(src)
            base=s[0]
            suf=s[-1]
            while True:
                newfilename=os.path.join(dest, f'{base}_{idx}{suf}')
                if os.path.isfile(newfilename):
                    idx+=1
                    continue
                shutil.move(src, newfilename)
                log.info(f'moved {src}->{newfilename}')
                break

    folder = os.path.join(SRC_DATA_FOLDER, '..')
    os.chdir(folder)
    print_help()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 800)

    def clamp(val,minval, maxval):
        if val < minval: return minval
        if val > maxval: return maxval
        return val

    while True:
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        if len(folder) == 0:
            log.info('aborted')
            quit(1)
        os.chdir(folder)
        ls = os.listdir()
        ls = sorted(ls)
        nfiles = len(ls)
        mode = 'fwd'
        idx = -1

        try:
            with tqdm(total=nfiles, file=sys.stdout) as pbar:
                while True:
                    idx = idx + (1 if mode == 'fwd' or mode == 'step-fwd' else -1)
                    pbar.update(idx)
                    if idx >= len(ls):
                        raise GetOutOfLoop
                    image_file_path = ls[idx]
                    if os.path.isdir(image_file_path):
                        continue
                    if not os.path.isfile(image_file_path):
                        log.warning(f'{image_file_path} missing, continuing with next image')
                        continue
                    try:
                        try:
                            color_mode = 'grayscale' if input_details[0]['shape'][3] == 1 else 'rgb'
                            img = tf.keras.preprocessing.image.load_img(image_file_path, color_mode=color_mode)
                        except PIL.UnidentifiedImageError as e:
                            log.warning(f'{e}: {image_file_path} is not an image?')
                            continue
                        if (img.format is None and (img.width != IMSIZE or img.height != IMSIZE)) and img.format != 'PNG' and img.format != 'JPEG':  # adobe media encoder does not set PNG type correctly
                            log.warning(f'{image_file_path} is not PNG or JPEG, skipping?')
                            continue
                        img_arr = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
                        img_arr = tf.image.resize(img_arr, (IMSIZE, IMSIZE))  # note we do NOT want black borders and to preserve aspect ratio! We just want to squash to square
                        is_joker, joker_prob, pred = classify_joker_img(img_arr, interpreter, input_details, output_details)
                        threshold_pause_prob = .05
                        # override default threshold to show ambiguous samples
                        file = jokers_list_file if is_joker else nonjokers_list_file
                        file.write(os.path.realpath(image_file_path) + '\n')
                        is_joker = joker_prob > JOKER_DETECT_THRESHOLD_SCORE
                        if not args.show_only_jokers or (args.show_only_jokers and is_joker):
                            img_arr = np.array(img_arr, dtype=np.uint8)  # make sure it is an np.array, not EagerTensor that cv2 cannot display
                            cv2.putText(img_arr, f'{image_file_path}: {joker_prob * 100:4.1f}% joker', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                            # print('\a'q)  # beep on some terminals https://stackoverflow.com/questions/6537481/python-making-a-beep-noise
                            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                            cv2.imshow('frame', img_arr)
                        d=15 # ms
                        if is_joker and pause_mode==1:
                            d=0
                        elif (joker_prob > threshold_pause_prob and pause_mode==0) or mode!='fwd':
                            d=0
                        k = cv2.waitKey(d)  # wait longer for joker detected
                        k = k & 0xff
                        if k == 27 or k == ord('q') or k == ord('x'):  # quit
                            cv2.destroyAllWindows()
                            jokers_list_file.close()
                            nonjokers_list_file.close()
                            log.info(f'jokers saved as {os.path.realpath(jokers_list_file.name)} and nonjokers saved in {os.path.realpath(nonjokers_list_file.name)}')
                            quit()
                        elif k == ord('.'):
                            mode = 'step-fwd'
                            continue
                        elif k == ord(','):
                            mode = 'step-back'
                            continue
                        elif k == ord('\n') or k == ord('\r'):  # enter/newline/cr
                            os.chdir('..')
                            raise GetOutOfLoop  # choose new folder
                        elif k == ord('h'):
                            print_help()
                        elif k == ord('t'):
                            pause_mode += 1
                            if pause_mode > 2:
                                pause_mode = 0
                            print(f'pause mode: {pause_modes[pause_mode]}')
                        elif k == ord('j'):
                            log.info(f'moving {image_file_path} to {JOKERS_FOLDER}')
                            try:
                                move_with_backup(image_file_path, JOKERS_FOLDER)
                            except Exception as e:
                                log.error(f'could not move {image_file_path}->{JOKERS_FOLDER}: caught {e}')

                        elif k==ord('g'):
                            idx=clamp(simpledialog.askinteger('go to frame', 'frame number?'),0,nfiles-1)
                        elif k==ord('f'):
                            idx=clamp(idx+100,0,nfiles-1)
                            log.info('fastforward')
                        elif k==ord('r'):
                            idx = clamp(idx - 100, 0, nfiles-1)
                            log.info('fastforward')
                        elif k == ord('n'):
                            log.info(f'moving {image_file_path} to {NONJOKERS_FOLDER}')
                            try:
                                move_with_backup(image_file_path, NONJOKERS_FOLDER)
                            except Exception as e:
                                log.error(f'could not move {image_file_path}->{NONJOKERS_FOLDER}: caught {e}')

                        elif k == 255 or k == ord(' '):  # no key or space
                            mode = 'fwd'
                            continue
                    except Exception as e:
                        log.error(f'caught {e} for file {image_file_path}')
                        raise GetOutOfLoop
        except GetOutOfLoop:
            log.info(f'jokers saved as {os.path.realpath(jokers_list_file.name)} and nonjokers saved in {os.path.realpath(nonjokers_list_file.name)}')
            continue

    jokers_list_file.close()
    nonjokers_list_file.close()


def test_random_samples():
    """ Runs test on test folder to evaluate accuracy on example images
    """
    import tensorflow as tf
    import cv2
    log.info('evaluating test set accuracy')
    log.info(f'Tensorflow version {tf.version.VERSION}')
    model = load_latest_model()

    # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert
    # existing_models = glob.glob(JOKER_NET_BASE_NAME + '_*.tflite')
    # latest_model = max(existing_models, key=os.path.getmtime)
    # log.info(f'loading latest tflite model {latest_model}')
    # time.sleep(5)
    # interpreter = tf.lite.Interpreter(model_path=latest_model)
    # interpreter.allocate_tensors()
    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    test_folder = TRAIN_DATA_FOLDER + '/valid/'
    import random
    ls = []
    class_folder_name = []
    idx = [0, 0]
    for c in [1, 2]:
        class_folder_name.append(test_folder + f'class{c}')
        ls.append(os.listdir(class_folder_name[c - 1]))
        random.shuffle(ls[c - 1])
    while True:
        windows = []
        for c in [1, 2]:
            gt_class = 'nonjoker' if c == 1 else 'joker'
            image_file_name = ls[c - 1][idx[c - 1]]
            image_file_path = os.path.join(class_folder_name[c - 1], image_file_name)
            img = tf.keras.preprocessing.image.load_img(image_file_path, color_mode='grayscale')
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = (1. / 255) * np.array([input_arr])  # Convert single image to a batch.
            pred = model.predict(input_arr)

            # img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (IMSIZE, IMSIZE))
            # input = (1. / 255) * np.array(np.reshape(img, [IMSIZE, IMSIZE, 1]),dtype=np.float32)
            # pred1 = model.predict(input[None,:])
            # input=(1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]))
            # pred = model.predict(input)
            # interpreter.set_tensor(input_details[0]['index'], (1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32))
            # interpreter.invoke()
            # pred = interpreter.get_tensor(output_details[0]['index'])

            dec = 'joker' if np.argmax(pred[0]) == 1 else 'nonjoker'
            joker_prob = pred[0][1]
            correct = 'right' if ((dec == 'joker' and c == 2) or (dec == 'nonjoker' and c == 1)) else 'wrong'
            win_name = f'{correct}: Real class:class{c}/{gt_class} detected as {dec} (joker_prob={joker_prob:.2f})'
            if correct == 'wrong':  # save wrong classifications for later
                copy_folder = TRAIN_DATA_FOLDER + '/incorrect/' + f'class{c}'
                Path(copy_folder).mkdir(parents=True, exist_ok=True)
                log.info(f'saving file {image_file_name} as incorrect {gt_class} classified as {dec}')
                copyfile(image_file_path, os.path.join(copy_folder, image_file_name))
            print(f'{image_file_path} {win_name}')
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            windows.append(win_name)
            cv2.imshow(win_name, np.array(img))
            cv2.moveWindow(win_name, 1, 500 * (c - 1) + 1)
            cv2.resizeWindow(win_name, 800, 400)

            if correct == 'wrong':
                k = cv2.waitKey(0) & 0xff
            else:
                k = cv2.waitKey(500) & 0xff
            if k == 27 or k == ord('q') or k == ord('x'):
                cv2.destroyAllWindows()
                quit()
            cv2.destroyWindow(win_name)
            idx[c - 1] += 1
            if idx[c - 1] >= len(ls[c - 1]):
                idx[c - 1] = 0


def classify_joker_img(img: np.array, interpreter, input_details, output_details, model:tf.keras.Model=None):
    """ Classify uint8 img

    :param img: input image as unit8 np.array
    :param interpreter: the TFLITE interpreter
    :param input_details: the input details of interpreter
    :param output_details: the output details of interpreter
    :param model: optional Keras Model; determines preprocessing for some models

    :returns: is_joker (True/False), joker_probability (0-1), prediction[2]=[nonjoker, joker]
    """
    nchan = input_details[0]['shape'][3]
    if model is not None and model.name=='mobilenet-roshambo':
        img = tf.keras.applications.mobilenet.preprocess_input(img)
        inp =np.reshape(img, [1, IMSIZE, IMSIZE, nchan])
    else:
        inp = (1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, nchan]), dtype=np.float32)  # todo not sure about 1/255 if mobilenet has input preprocessing with uint8 input
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    pred_vector = interpreter.get_tensor(output_details[0]['index'])[0]
    joker_prob = pred_vector[1]
    is_joker = pred_vector[1] > pred_vector[0] and joker_prob > JOKER_DETECT_THRESHOLD_SCORE
    return is_joker, joker_prob, pred_vector


def load_latest_model():
    existing_model_folders = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '*/')
    model = None

    if len(existing_model_folders) > 0:
        log.info(f'found existing models:\n{existing_model_folders}\n choosing newest one')
        latest_model_folder = max(existing_model_folders, key=os.path.getmtime)
        log.info(f'*** initializing model from {latest_model_folder}')
        time.sleep(3)
        model = load_model(latest_model_folder)
        # model.compile()
        model.summary()
        print(f'model.input_shape: {model.input_shape}')
    else:
        log.error('no model found to load')
        quit(1)
    return model


def load_tflite_model(folder=None, dialog=True):
    """ loads the most recent trained TFLITE model

    :param folder: folder where TFLITE_FILE_NAME is to be found, or None to find latest one
    :param dialog: set False to raise FileNotFoundError or True to open file dialog to browse for model
    :returns: interpreter,input_details,output_details

    :raises: FileNotFoundError if TFLITE_FILE_NAME is not found in folder
    """
    tflite_model_path = None
    if folder is None:
        existing_models = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '_*/')
        if len(existing_models) > 0:
            latest_model_folder = max(existing_models, key=os.path.getmtime)
            tflite_model_path = os.path.join(latest_model_folder, TFLITE_FILE_NAME)
            if not os.path.isfile(tflite_model_path):
                if not dialog:
                    raise FileNotFoundError(f'no TFLITE model found at {tflite_model_path}')
                else:
                    root = Tk()
                    root.withdraw()
                    tflite_model_path = filedialog.askopenfilename(initialdir= MODEL_DIR, title='Select TFLITE model', filetypes=[('Tensorflow lite models','*.tflite')])
                    if tflite_model_path is None:
                        log.info('aborted')
                        quit(1)
        else:
            raise FileNotFoundError(f'no models found in {MODEL_DIR}')
    else:
        tflite_model_path = os.path.join(folder, TFLITE_FILE_NAME)
    log.info('loading tflite CNN model {}'.format(tflite_model_path))
    # model = load_model(MODEL)
    # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def measure_flops():
    log.info('measuring Op/frame for CNN')
    from flopco_keras import FlopCoKeras

    model = create_model(ask_for_comment=False)  # load_latest_model() #tf.keras.applications.ResNet101()
    flopco = FlopCoKeras(model)

    # log.info(f'flop counter: {str(flopco)}')
    log.info(f"Op/frame: {eng(flopco.total_flops)}")
    log.info(f"MAC/frame: {eng(flopco.total_macs)}")
    s = 'Fractional Op per layer: '
    for f in flopco.relative_flops:
        s = s + f' {f * 100:.2f}%'
    log.info(f'Fractional Op per layer: {s}')

    return flopco

    # session = tf.compat.v1.Session()
    # graph = tf.compat.v1.get_default_graph()
    #
    # with graph.as_default():
    #     with session.as_default():
    #         model=load_latest_model()
    #         run_meta = tf.compat.v1.RunMetadata()
    #         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    #
    #         image_file_path = os.path.join('data/joker.png')
    #         img = tf.keras.preprocessing.image.load_img(image_file_path, color_mode='grayscale')
    #         input_arr = tf.keras.preprocessing.image.img_to_array(img)
    #         input_arr = (1. / 255) * np.array([input_arr])  # Convert single image to a batch.
    #         pred = model.predict(input_arr)
    #
    #
    #         # Optional: save printed results to file
    #         flops_log_path = os.path.join('.', 'tf_flops_log.txt')
    #         opts['output'] = 'file:outfile={}'.format(flops_log_path)
    #
    #         # We use the Keras session graph in the call to the profiler.
    #         flops = tf.compat.v1.profiler.profile(graph=graph,
    #                                               run_meta=run_meta, cmd='op', options=opts)
    #
    #         return flops.total_float_ops


def create_model_alexnet():
    """ Creates the CNN model for joker detection
    """
    model = Sequential()

    # model.add(Input(shape=(None,IMSIZE,IMSIZE,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=128, kernel_size=(7, 7),
                     strides=(3, 3), padding='valid',
                     input_shape=(IMSIZE, IMSIZE, 1),
                     activation='relu', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv3'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv4'))
    # model.add(Conv2D(filters=256, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(30, activation='relu', name='fc1'))
    model.add(Dense(30, activation='relu', name='fc2'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(2, activation='softmax', name='output'))

    return model


def create_model_resnet():
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(IMSIZE, IMSIZE, 1), weights=None, include_top=False)
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tensorflow.keras.layers.Dense(2, activation='softmax')(x)
    model = tensorflow.keras.models.Model(inputs=[base_model.input], outputs=[output])
    # train
    # model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_model_mobilenet():
    """ Creates MobileNet model
    https://keras.io/api/applications/mobilenet/

    """
    weights = None # 'imagenet'
    alpha = .25  # 0.25 is smallest version with imagenet weights
    depth_multiplier = int(1)
    dropout = 0.5  # default is .001
    include_top = True  # set false to specify our own FC layers
    fully_connected_layers = (128, 128)  # our FC layers
    num_input_channels = 3 if weights == 'imagenet' else 1
    pooling = 'avg'  # default is avg for mobilenet
    freeze = False  # set true to freeze imagenet feature weights
    log.info(f'creating MobileNet with weights={weights} alpha={alpha} depth_multiplier={depth_multiplier} dropout={dropout} fully_connected_layers={fully_connected_layers} pooling={pooling} frozen_imagenet_layers={freeze}')
    model = tf.keras.applications.MobileNet(
        input_shape=(IMSIZE, IMSIZE, num_input_channels),  # must be 3 channel input if using imagenet weights
        weights=weights,
        include_top=include_top,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        dropout=dropout,
        input_tensor=None,
        pooling=pooling,
        classes=2,
        classifier_activation="softmax",
    )
    model.trainable = not freeze

    if not include_top:  # if we add our own FC output, then we need to wrap it with input layers
        # x=Flatten()(model.output)
        x = GlobalAveragePooling2D()(model.output)
        for n in fully_connected_layers:
            x = Dense(n, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
        # Output Layer
        output = Dense(2, activation='softmax', name='output')(x)
        final = Model(inputs=model.inputs, outputs=output, name='joker-mobilenet')
        return final
    model.summary(print_fn=log.info)
    return model


def create_model(ask_for_comment=True):
    """
    Creates a new instance of model, returning the model folder, and starts file logger in it

    :param ask_for_comment: True to ask for comment as part of filename, also creates a folder for the model. Set to False to just create model in memory.

    :returns: model, model_folder_path
    """
    model_type = create_model_mobilenet
    new_model_folder_name = None
    if ask_for_comment:
        model_comment = input('short model comment?')
        model_comment.replace(' ', '_')

        start_timestr = time.strftime("%Y%m%d-%H%M")
        mcs = f'_{model_comment}' if model_comment is not None else ''
        new_model_folder_name = os.path.join(MODEL_DIR, f'{JOKER_NET_BASE_NAME}{mcs}_{start_timestr}')
        Path(new_model_folder_name).mkdir(parents=True, exist_ok=True)
    log.info(f'creating new empty model at {new_model_folder_name}')
    return model_type(), new_model_folder_name


def measure_latency(model_folder=None):
    log.info('measuring CNN latency in loop')
    interpreter, input_details, output_details = load_tflite_model(model_folder)
    nchan = input_details[0]['shape'][3]
    img = np.random.randint(0, 255, (IMSIZE, IMSIZE, nchan))
    N = 100
    for i in range(1, N):
        with Timer('CNN latency') as timer:
            classify_joker_img(img, interpreter, input_details, output_details)
    timer.print_timing_info(log)


class PlotHistoryCallback(History):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        keys = list(logs.keys())
        log.info("End epoch {} of training; got log keys: {}".format(epoch, keys))
        plot_history(self.model.history, 'history')


class SGDLearningRateTracker(Callback):

    def on_epoch_begin(self, batch, logs=None):
        try:
            optimizer = self.model.optimizer
            current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            log.info(f'SGD learning rate: {current_decayed_lr:.6f}')
        except Exception as e:
            log.error(f'caught {e} trying to get current learning rate')


def train(args=None):
    start_time = time.time()
    start_timestr = time.strftime("%Y%m%d-%H%M")

    checkpoint_filename_path = os.path.join(MODEL_DIR, 'joker_net_checkpoint.hdf5')

    model = None
    model_folder = None

    if INITIALIZE_MODEL_FROM_LATEST:
        existing_model_folders = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '*/')
        TIMEOUT = 30
        if len(existing_model_folders) > 0:
            model_folder = max(existing_model_folders, key=os.path.getmtime)
            getmtime_checkpoint = os.path.getmtime(checkpoint_filename_path) if os.path.isfile(checkpoint_filename_path) else 0
            getmtime_stored_model = os.path.getmtime(model_folder)
            if os.path.isfile(checkpoint_filename_path) \
                    and getmtime_checkpoint > getmtime_stored_model \
                    and yes_or_no(f'checkpoint {checkpoint_filename_path} modified {datetime.datetime.fromtimestamp(getmtime_checkpoint)} \nis newer than saved model {existing_model_folders[0]} modified {datetime.datetime.fromtimestamp(getmtime_stored_model)},\n start from it?', timeout=TIMEOUT):
                log.info(f'loading weights from checkpoint {checkpoint_filename_path}')
                model, model_folder = create_model()
                model.load_weights(checkpoint_filename_path)
            else:
                if yes_or_no(f'model {model_folder} exists, start from it?', timeout=TIMEOUT):
                    log.info(f'initializing model from {model_folder}')
                    model = load_model(model_folder)
                else:
                    model, model_folder = create_model()
        else:
            if os.path.isfile(checkpoint_filename_path) and yes_or_no('checkpoint exists, start from it?', timeout=TIMEOUT):
                model, model_folder = create_model()
                log.info(f'loading weights from checkpoint {checkpoint_filename_path}')
                model.load_weights(checkpoint_filename_path)
            else:
                yn = yes_or_no("Could not find saved model or checkpoint. Initialize a new model?", timeout=TIMEOUT)
                if yn:
                    model, model_folder = create_model()
                else:
                    log.warning('aborting training')
                    quit(1)
    else:
        model, model_folder = create_model()

    LOG_FILE = os.path.join(model_folder, f'training-{start_timestr}.log')
    fh = logging.FileHandler(LOG_FILE, 'w')  # 'w' to overwrite, not append
    fh.setLevel(logging.INFO)
    fmtter = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(message)s")
    fh.setFormatter(fmtter)
    log.addHandler(fh)
    log.info(f'added logging handler to {LOG_FILE}')

    log.info(f'Tensorflow version {tf.version.VERSION}')
    log.info(f'dataset path: TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}')
    log.info(f'TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}\nSRC_DATA_FOLDER={SRC_DATA_FOLDER}')
    log.debug('test debug')
    log.warning('test warning')
    log.error('test error')

    train_batch_size = 32
    valid_batch_size = 64
    test_batch_size = 64

    color_mode = 'grayscale' if model.input_shape[3] == 1 else 'rgb'
    train_datagen = ImageDataGenerator(  # 实例化
        rescale=1. / 255,  # todo check this
        rotation_range=15,  # 图片随机转动的角度
        width_shift_range=0.2,  # 图片水平偏移的幅度
        height_shift_range=0.2,  # don't shift too much vertically to avoid losing top of card
        fill_mode='constant',
        cval=0,  # fill edge pixels with black; default fills with long lines of color
        zoom_range=[.9, 1.25],  # NOTE zoom >1 minifies, don't zoom in (<1) too much in to avoid losing joker part of card
        # horizontal_flip=False,
    )  # 随机放大或缩小

    log.info('making training generator')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/train/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        # save_to_dir='/tmp/augmented_images',save_prefix='aug', # creates zillions of samples, watch out! make the folder before running or it will not work
        shuffle=True,
        interpolation='nearest',
    )

    log.info('making validation generator')

    valid_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/valid/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True,  # irrelevant for validtion
        interpolation='nearest',
    )

    log.info('making test generator')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(  # 实例化
        TRAIN_DATA_FOLDER + '/test/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False,
        interpolation='nearest',
    )  # IMPORTANT shuffle=False here or model.predict will NOT match GT of test generator in test_generator.labels!

    # Path('test_gen_samples').mkdir(parents=True, exist_ok=True)

    def print_datagen_summary(gen: ImageDataGenerator):
        nsamp = gen.samples
        num_nonjoker, num_joker = np.bincount(gen.labels)

        def pc(n):
            return 100 * float(n) / nsamp

        log.info(f'summary of {gen.directory}:'
                 f' {gen.samples} samples:\t{num_nonjoker}/{pc(num_nonjoker):.3f}% nonjoker,\t{num_joker}/{pc(num_joker):.3f}% joker\n')


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=train_generator.n // train_batch_size,  # every epoch decrease learning rate
        decay_rate=0.9)


    optimizer = tf.keras.optimizers.SGD(momentum=.9, learning_rate=lr_schedule)  # alessandro: SGD gives higher accuracy than Adam but include a momentuum
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary(print_fn=log.info)

    stop_early = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
    save_checkpoint = ModelCheckpoint(checkpoint_filename_path, save_best_only=True, monitor='val_loss', mode='min')
    # Profile from batches 10 to 15
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tb_profiler_log', profile_batch='10, 15')
    plot_callback = PlotHistoryCallback()
    learning_rate_callback = SGDLearningRateTracker()

    if args.train:
        log.info('starting training')
        epochs = 300

        class_weight = {0: .5, 1: .5}  # not sure about this weighting. should we weight the nonjoker more heavily to avoid false positive jokers? The ratio is about 4:1 nonjoker/joker samples.
        log.info(f'Training uses class_weight={class_weight} (nonjoker/joker) with max {epochs} epochs optimizer={optimizer} and train_batch_size={train_batch_size} ')
        history = None
        log.info(f'learning rate schedule: {lr_schedule}')
        print_datagen_summary(train_generator)
        print_datagen_summary(valid_generator)
        print_datagen_summary(test_generator)
        try:
            history = model.fit(train_generator,  # todo back to train generator
                                # steps_per_epoch=150,
                                epochs=epochs, verbose=1,
                                validation_data=valid_generator,
                                # validation_steps=25,
                                callbacks=[stop_early, save_checkpoint, plot_callback, learning_rate_callback],
                                max_queue_size=20,
                                use_multiprocessing=False,
                                # shuffle = True, # shuffle not valid for folder data generators
                                workers=8,
                                class_weight=class_weight  # weight nonjokers more to reduce false positives where nonjokers are detected as jokers poking the finger out
                                # it is better to avoid poking out finger until a joker is definitely detected
                                )
            plot_history(history, start_timestr)
            training_history_filename = os.path.join(LOG_DIR, 'training_history' + '-' + start_timestr + '.npy')
            np.save(training_history_filename, history.history)
            log.info(f'Done with model.fit; history is \n{history.history} and is saved as {training_history_filename}')
            log.info(f'history.history.keys()={history.history.keys()}')

        except KeyboardInterrupt:
            log.warning('keyboard interrupt, saving model and testing')

        log.info(f'saving model to folder {model_folder}')
        model.save(model_folder)

        log.info('converting model to tensorflow lite model')
        converter = tf.lite.TFLiteConverter.from_saved_model(model_folder)  # path to the SavedModel directory
        tflite_model = converter.convert()
        tflite_model_path = os.path.join(model_folder, TFLITE_FILE_NAME)

        log.info(f'saving tflite model as {tflite_model_path}')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

    log.info('evaluating accuracy')
    # test_generator.reset()
    gen = test_generator
    gen.reset()
    loss, acc = model.evaluate(gen, verbose=1)
    log.info(f'On test set {gen.directory}  loss={loss:.3f}, acc={acc:.4f}')
    gen.reset()
    y_output = model.predict(gen, verbose=1)  # matrix of Nx2 with each row being the nonjoker/joker score
    y_pred = np.argmax(y_output, axis=1)  # vector of predicted classes 0/1 for nonjoker/joker
    y_true = gen.labels  # vector of ground truth classes, 0 or 1 for nonjoker/joker
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    log.info(f'**** final test set balanced accuracy: {balanced_accuracy * 100:6.3f}% (chance would be 50%)\nConfusion matrix nonjoker/joker:\n {conf_matrix}')
    np.set_printoptions(precision=2)
    # disp = plot_confusion_matrix(classifier, X_test, y_test,
    #                                  display_labels=['nonjoker','joker'],
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=True)
    # disp.ax_.set_title('joker/nonjoker confusion matrix')

    try:
        measure_flops()
    except Exception as e:
        log.error(f'Caught {e} measuring flops')
    measure_latency(model_folder)
    elapsed_time_min = (time.time() - start_time) / 60
    if args.train:
        log.info(f'**** done training after {elapsed_time_min:4.1f}m; model saved in {model_folder}.'
                 f'\nSee {LOG_FILE} for logging output for this run.')


def plot_history(history, start_timestr):
    if history is not None:
        try:
            # TODO make subplots and make axis log with times in ms, not log ms
            # summarize history for accuracy
            plt.figure('accuracy')
            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig(os.path.join(LOG_DIR, 'accuracy' + '-' + start_timestr + '.png'))
            plt.show()
            # summarize history for loss
            plt.figure('loss')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig(os.path.join(LOG_DIR, 'loss' + '-' + start_timestr + '.png'))
            plt.show()
        except KeyError as k:
            log.warning('could not plot, caught {k}, history.history.keys()={history.history.keys()} ')


def generate_augmented_images(args):
    folder = TRAIN_DATA_FOLDER
    os.chdir(folder)
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    if len(folder) == 0:
        log.info('aborted')
        quit(1)
    os.chdir(folder)

    train_datagen = ImageDataGenerator(  # 实例化
        rescale=1. / 255,  # todo check this
        rotation_range=15,  # 图片随机转动的角度
        width_shift_range=0.2,  # 图片水平偏移的幅度
        height_shift_range=0.2,  # don't shift too much vertically to avoid losing top of card
        fill_mode='constant',
        cval=0,  # fill edge pixels with black; default fills with long lines of color
        zoom_range=[.9, 1.25],  # NOTE zoom >1 minifies, don't zoom in (<1) too much in to avoid losing joker part of card
        # horizontal_flip=False,
    )  # 随机放大或缩小

    log.info('making training generator')
    train_generator = train_datagen.flow_from_directory(
        '.',
        target_size=(IMSIZE, IMSIZE),
        batch_size=8,  # small batch
        class_mode='categorical',
        color_mode='grayscale',
        # save_to_dir='/tmp/augmented_images',save_prefix='aug', # creates zillions of samples, watch out! make the folder before running or it will not work
        shuffle=True,
        interpolation='nearest',
    )

    log.info(f'train_datagen: {train_datagen}')
    log.info(f'train_generator: {train_generator.n} images')
    folder = os.path.join(SRC_DATA_FOLDER, 'augmented')
    Path(folder).mkdir(parents=True, exist_ok=True)

    cv2.namedWindow('nonjoker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('nonjoker', 800, 800)
    cv2.namedWindow('joker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('joker', 800, 800)
    while True:
        batch = train_generator.__next__()
        imgs = batch[0]
        labels = batch[1]
        for i in range(imgs.shape[0]):
            if labels[i][0] > labels[i][1]:
                cv2.imshow('nonjoker', imgs[i])
            else:
                cv2.imshow('joker', imgs[i])

            k = cv2.waitKey(15) & 0xff
            if k == ord('x') or k == ord('q'):
                quit(0)


args = None  # if run as module

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train:train CNN for trixsy', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", action='store_true', help="train model starting from latest (shows options to choose whether to initialize.")
    parser.add_argument("--test_accuracy", action='store_true', help="run network test rather than train.")
    parser.add_argument("--riffle_test", action='store_true', help="test by pausing playback/classify of recorded frames at detected jokers.")
    parser.add_argument("--show_only_jokers", action='store_true', help="for riffle_test, show only detected jokers in cv2 windows.")
    parser.add_argument("--rename_images", action='store_true', help="rename images in a folder consecutively sorting them by mtime.")
    parser.add_argument("--make_training_set", action='store_true', help="make training data from source images.")
    parser.add_argument("--test_random_samples", action='store_true', help="test random samples from test set.")
    parser.add_argument("--measure_flops", action='store_true', help="measures flops/frame of network.")
    parser.add_argument("--measure_latency", action='store_true', help="measures CNN latency.")
    parser.add_argument("--augment_training_set", action='store_true', help="show augmented training images using default augmentation.")

    args = parser.parse_args()
    if args.train or args.test_accuracy:
        train(args)
    elif args.rename_images:
        rename_images()
    elif args.make_training_set:
        make_training_set()
    elif args.test_random_samples:
        test_random_samples()
    elif args.riffle_test:
        riffle_test(args)
    elif args.measure_latency:
        measure_latency()
    elif args.augment_training_set:
        generate_augmented_images(args)
    elif args.measure_flops:
        stats = measure_flops()
        log.info(f'total flops/frame={eng(stats.total_flops)}')
    else:
        parser.print_help()
