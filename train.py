# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train/ valid/ test/ folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
# author: Tobi Delbruck
import argparse
import collections
import datetime
import glob
import shutil
import tempfile
import random
from shutil import copyfile
from tkinter import *
from tkinter import filedialog, simpledialog

# uncomment lines to run on CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import PIL
import tensorflow.python.keras
# from classification_models.keras import Classifiers
import tensorflow_addons as tfa # used for focalloss
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
# from alessandro: use keras from tensorflow, not from keras directly
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tqdm import tqdm

from globals_and_utils import *


INITIALIZE_MODEL_FROM_LATEST = True  # set True to initialize weights to latest saved model

import logging
import sys

log = my_logger(__name__)


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def rename_images(folder=SRC_DATA_FOLDER):
    """ Cleans up a folder filled with images (png and jpg) so that the images are numbered consecutively. Useful after using mv --backup=t to add new images to a folder
    :param folder: the folder name to clean up, relative to working directory
    """
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
              'u undues last move of joker/nonjoker\n'
              't toggles between pausing at possible joker|pausing only at certain jokers|not pausing\n'
              'enter selects new playback folder\n'
              'h print help')

    pause_modes={0:'pause_possible',1:'pause_certain',2:'dont_pause'}
    pause_mode=0 # which classifications to pause for

    model,interpreter, input_details, output_details = load_latest_model(dialog=False)
    Path(JOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(NONJOKERS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    jokers_list_file = open(os.path.join(LOG_DIR, f'joker-file-list-{start_timestr}.txt'), 'w')
    nonjokers_list_file = open(os.path.join(LOG_DIR, f'nonjoker-file-list-{start_timestr}.txt'), 'w')
    undo_list=collections.deque(maxlen=5)

    def move_with_backup(src, dest_folder):
        if not os.path.isdir(dest_folder):
            raise Exception(f'destination {dest_folder} should be a directory')
        src_base=os.path.split(src)[-1]
        dest_file_path=os.path.join(dest_folder, src_base)
        try:
            shutil.move(src, dest_file_path)
            undo_list.append((src,dest_file_path))
            log.info(f'moved {src}->{dest_file_path}')
        except OSError as e:
            log.info(f'{dest_file_path} already exists, moving to a new filename...')
            idx=1
            s=os.path.splitext(src)
            src_base=s[0]
            suf=s[-1]
            while True:
                newfilename=os.path.join(dest_folder, f'{src_base}_{idx}{suf}')
                if os.path.isfile(newfilename):
                    log.info(f'{newfilename} already exists...')
                    idx+=1
                    continue
                shutil.move(src, newfilename)
                log.info(f'moved {src}->{newfilename}')
                undo_list.append((src,newfilename))
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

    def undo():
        src=None
        dest=None
        try:
            src,dest=undo_list.pop()
            shutil.move(dest,src)
            log.info(f'undid by moving {dest}->{src}')
        except IndexError:
            log.warning('nothing to undo')
        except OSError as e:
            log.error(f'could not undo by moving {src}->{dest}: caught {e}')

    while True:
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        if len(folder) == 0:
            log.info('aborted')
            quit(1)
        os.chdir(folder)
        log.info(f'listing contents of {folder} ')
        ls = os.listdir()
        nfiles = len(ls)
        log.info(f'sorting {nfiles} samples by date')
        ls = sorted(ls,key=lambda f: os.stat(f).st_mtime)
        mode = 'fwd'
        idx = -1

        try:
            with tqdm(total=nfiles, file=sys.stdout) as pbar:
                while True:
                    idx = idx + (1 if mode == 'fwd' or mode == 'step-fwd' else -1)
                    if idx<0:
                        idx=len(ls)-1
                        time.sleep(.5)
                    if idx >= len(ls):
                        idx=0
                        time.sleep(.5)
                    pbar.update(idx)
                    image_file_path = ls[idx]
                    if os.path.isdir(image_file_path):
                        continue
                    if not os.path.isfile(image_file_path):
                        log.warning(f'{image_file_path} missing, continuing with next image')
                        continue
                    try:
                        try:
                            img_arr=cv2.imread(image_file_path,cv2.IMREAD_GRAYSCALE)
                            img_arr=(1/255.)* np.array(img_arr)
                        except Exception as e2:
                            log.warning(f'{e2}: {image_file_path} is not an image?')
                            continue
                        if img_arr.shape[0]!= IMSIZE or img_arr.shape[1] != IMSIZE:
                            log.warning(f'{image_file_path} has wrong shape {img_arr.shape}, skipping')
                            continue
                        # try:
                        #     img = tf.keras.preprocessing.image.load_img(image_file_path, color_mode='grayscale')
                        # except PIL.UnidentifiedImageError as e:
                        #     log.warning(f'{e}: {image_file_path} is not an image?')
                        #     continue
                        # if (img.format is None and (img.width != IMSIZE or img.height != IMSIZE)) and img.format != 'PNG' and img.format != 'JPEG':  # adobe media encoder does not set PNG type correctly
                        #     log.warning(f'{image_file_path} is not PNG or JPEG, skipping?')
                        #     continue
                        # img_arr = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
                        # img_arr = tf.image.resize(img_arr, (IMSIZE, IMSIZE))  # note we do NOT want black borders and to preserve aspect ratio! We just want to squash to square
                        is_joker, joker_prob, pred = classify_joker_img(img_arr, model, interpreter, input_details, output_details)
                        threshold_pause_prob = .05
                        # override default threshold to show ambiguous samples
                        file = jokers_list_file if is_joker else nonjokers_list_file
                        file.write(os.path.realpath(image_file_path) + '\n')
                        is_joker = joker_prob > JOKER_DETECT_THRESHOLD_SCORE
                        if not args.show_only_jokers or (args.show_only_jokers and is_joker):
                            # cv2_arr = np.array(img_arr, dtype=np.uint8)  # make sure it is an np.array, not EagerTensor that cv2 cannot display
                            cv2_arr = 1-img_arr  # make sure it is an np.array, not EagerTensor that cv2 cannot display
                            cv2.putText(cv2_arr, f'{image_file_path}: {joker_prob * 100:4.1f}% joker', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                            # print('\a'q)  # beep on some terminals https://stackoverflow.com/questions/6537481/python-making-a-beep-noise
                            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                            cv2.imshow('frame', cv2_arr)
                        d=50 # ms
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
                        elif k == ord('u'):
                            undo()
                            idx-=1
                        elif k == ord('t'):
                            pause_mode += 1
                            if pause_mode > 2:
                                pause_mode = 0
                            print(f'pause mode: {pause_modes[pause_mode]}')
                        elif k == ord('j') or k==ord('n'):
                            dest_folder=JOKERS_FOLDER if k==ord('j') else NONJOKERS_FOLDER
                            src_file_path=os.path.join(folder,image_file_path)
                            log.info(f'moving {src_file_path} to {dest_folder}')
                            try:
                                move_with_backup(src_file_path, dest_folder)
                            except Exception as e:
                                log.error(f'could not move {image_file_path}->{JOKERS_FOLDER}: caught {e}')
                        elif k==ord('g'):
                            idx=clamp(simpledialog.askinteger('go to frame', 'frame  number?'),0,nfiles-1)
                        elif k==ord('f'):
                            idx=clamp(idx+50,0,nfiles-1)
                            log.info('fastforward')
                        elif k==ord('r'):
                            idx = clamp(idx - 50, 0, nfiles-1)
                            log.info('fastforward')
                        elif k == ord(' '):  # space
                            mode='fwd'
                        elif k == 255:  # no key or space
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
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            pred = model.predict(input_arr)

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


def classify_joker_img(img: np.array, model:tf.keras.Model, interpreter, input_details, output_details):
    """ Classify img

    :param img: input image which should be float32 np.array with pixel values in range 0-1, i.e. 1./255 * uint8 array of original PNG
    :param interpreter: the TFLITE interpreter
    :param input_details: the input details of interpreter
    :param output_details: the output details of interpreter
    :param model: optional Keras Model; determines preprocessing for some models

    :returns: is_joker (True/False), joker_probability (0-1), prediction[2]=[nonjoker, joker]
    """
    is_joker=None
    joker_prob=None
    pred_vector=None
    # img2=np.array(img)
    # inp=np.array(img2[np.newaxis,...],dtype=np.float32)
    if USE_TFLITE:
        # nchan = input_details[0]['shape'][3]
        inp =  np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32)  # todo not sure about 1/255 if mobilenet has input preprocessing with uint8 input
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        pred_vector = interpreter.get_tensor(output_details[0]['index'])[0]
        joker_prob = pred_vector[1]
        is_joker = pred_vector[1] > pred_vector[0] and joker_prob > JOKER_DETECT_THRESHOLD_SCORE
    else: # use full TF model
        inp =  np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32)  # todo not sure about 1/255 if
        pred=model.predict(inp)
        pred_vector = pred[0]
        joker_prob = pred_vector[1]
        is_joker = pred_vector[1] > pred_vector[0] and joker_prob > JOKER_DETECT_THRESHOLD_SCORE

    return is_joker, joker_prob, pred_vector


def load_latest_model(folder=None, dialog=True, use_tflite_model=USE_TFLITE):
    """ Loads the latest trained model.  It loads the tensorflow 2 model or the tensorflow lite interpreter, depending on
    flag use_tflite_model
    along with input and output details for the TFLITE interpreter.

    :param folder: folder where TFLITE_FILE_NAME is to be found, or None to find latest one
    :param dialog: set False to raise FileNotFoundError or True to open file dialog to browse for model
    :param use_tflite_model: set False load normal TF mode, True to load TFLITE model

    :returns: model, tflite_interpreter, input_details, output_details, model_folder
    """
    # existing_model_folders = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '*/')
    model = None
    use_existing_model=True
    if dialog:
        use_existing_model=yes_or_no('use latest existing model?','y',timeout=10)
    if use_existing_model:
        tflite_model_path=None
        model_folder = None
        if folder is not None:
            model_folder=folder
            tflite_model_path = os.path.join(model_folder, TFLITE_FILE_NAME)
        else:
            paths = sorted(Path(MODEL_DIR).iterdir(), key=os.path.getmtime, reverse=True)
            # existing_models = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '_*/')
            for model_folder in paths:
                tflite_model_path = os.path.join(model_folder, TFLITE_FILE_NAME)
                if not os.path.isfile(tflite_model_path):
                    continue
                else:
                    break
        if model_folder is None:
            log.error(f'no folder found in {MODEL_DIR} with {TFLITE_FILE_NAME}')
            quit(1)
        log.info(f'loading CNN model from {model_folder}')
        if not use_tflite_model:
            model = load_model(model_folder)
            interpreter =  None
            # Get input and output tensors.
            input_details =None
            output_details = None
        else:
            # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert
            model=None
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
    else:
        model=create_model(ask_for_comment=False)
        model.compile()  # cannot save it at this point, TODO check what is needed to be able to save
        with tempfile.TemporaryDirectory() as model_folder:
            model.save(model_folder)
            log.info('converting model to tensorflow lite model')
            converter = tf.lite.TFLiteConverter.from_saved_model(model_folder)  # path to the SavedModel directory
            tflite_model = converter.convert()
            tflite_model_path = os.path.join(model_folder, TFLITE_FILE_NAME)

            log.info(f'saving tflite model as {tflite_model_path}')
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
                interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                interpreter.allocate_tensors()
                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

    return model,interpreter, input_details, output_details, model_folder

def measure_flops(model=None):
    log.info('measuring Op/frame for CNN')
    from flopco_keras import FlopCoKeras

    if model is None:
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
    do=.1
    l1reg=1e-2

    log.info(f'making LeNet with dropout={do} and L1 weight regularization with cost {l1reg}')

    model = Sequential()

    # model.add(Input(shape=(None,IMSIZE,IMSIZE,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=64, kernel_size=(19, 19),
                     strides=(3, 3), padding='valid',
                     input_shape=(IMSIZE, IMSIZE, 1),
                     activation='relu', name='conv1',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv2',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv3',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv4',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))
    # model.add(Conv2D(filters=256, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu', name='fc1',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))
    model.add(Dense(64, activation='relu', name='fc2',kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dropout(do))

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
    weights = None # for 'imagenet' must install classification_models package
    alpha = .5  # 0.25 is smallest version with imagenet weights
    depth_multiplier = int(1)
    dropout = 0.1  # default is .001
    fully_connected_layers = (128, 128)  # our FC layers
    pooling = 'max'  # default is avg for mobilenet
    log.info(f'creating MobileNet with weights={weights} alpha={alpha} depth_multiplier={depth_multiplier} dropout={dropout} fully_connected_layers={fully_connected_layers} pooling={pooling}')
    model = tf.keras.applications.MobileNet(
        input_shape=(IMSIZE, IMSIZE, 1),  # must be 3 channel input if using imagenet weights
        weights=weights,
        include_top=False,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        dropout=dropout,
        input_tensor=None,
        pooling=pooling,
        classes=2,
        classifier_activation="softmax",
    )
    model.trainable = True

    x=Flatten()(model.output)
    # x = GlobalAveragePooling2D()(model.output)
    for n in fully_connected_layers:
        x = Dense(n, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    # Output Layer
    output = Dense(2, activation='softmax', name='output')(x)
    final = Model(inputs=model.inputs, outputs=output, name='joker-mobilenet')
    return final


def create_model(ask_for_comment=True) ->tf.keras.Model:
    """
    Creates a new instance of model, returning the model folder, and starts file logger in it

    :param ask_for_comment: True to ask for comment as part of filename, also creates a folder for the model. Set to False to just create model in memory.

    :returns: model, model_folder_path
    """
    model_type = create_model_alexnet
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
    dialog=False
    if model_folder is None:
        dialog=True
    model, interpreter, input_details, output_details = load_latest_model(folder=model_folder,dialog=dialog)
    nchan = input_details[0]['shape'][3]
    img = np.random.randint(0, 255, (IMSIZE, IMSIZE, nchan))
    N = 100
    for i in range(1, N):
        with Timer('CNN latency') as timer:
            classify_joker_img(img, model, interpreter, input_details, output_details)
    timer.print_timing_info(log)


class PlotHistoryCallback(History):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        keys = list(logs.keys())
        log.info("End epoch {} of training; got log keys: {}".format(epoch, keys))
        s=f'End of epoch {epoch}: '
        for k in logs.keys():
            s+=f'{k}: {eng(logs[k])} '
        log.info(s)
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
    """ Train CNN

    :param args: use to specify if just to test. Set args.train=True if training wanted.

    """
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
    valid_batch_size = 128
    test_batch_size = 128
    train_datagen=None
    train_generator=None
    valid_generator=None
    test_generator=None

    augmentation_enabled=True

    color_mode = 'grayscale'
    train_datagen=None
    if args.train:
        if augmentation_enabled:
            train_datagen = ImageDataGenerator(  # 实例化
                rescale=1. / 255,  # we don't rescale, just leave 0-255
                rotation_range=15,  # 图片随机转动的角度
                width_shift_range=0.2,  # 图片水平偏移的幅度
                height_shift_range=0.2,  # don't shift too much vertically to avoid losing top of card
                fill_mode='constant',
                cval=0,  # fill edge pixels with black; default fills with long lines of color
                zoom_range=[.9, 1.25],  # NOTE zoom >1 minifies, don't zoom in (<1) too much in to avoid losing joker part of card
                # horizontal_flip=False,
            )  # 随机放大或缩小
        else:
            # train_datagen = ImageDataGenerator(rescale=1. / 255)
            train_datagen = ImageDataGenerator(rescale=1. / 255)

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

    # Path('test_gen_samples').mkdir(parents=True, exist_ok=True)

    def print_datagen_summary(gen: ImageDataGenerator):
        nsamp = gen.samples
        num_nonjoker, num_joker = np.bincount(gen.labels)

        def pc(n):
            return 100 * float(n) / nsamp

        log.info(f'summary of {gen.directory}:'
                 f' {gen.samples} samples:\t{num_nonjoker}/{pc(num_nonjoker):.3f}% nonjoker,\t{num_joker}/{pc(num_joker):.3f}% joker\n')



    if args.train:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=train_generator.n // train_batch_size,  # every epoch decrease learning rate
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(momentum=.9, learning_rate=lr_schedule)  # alessandro: SGD gives higher accuracy than Adam but include a momentuum
        loss = tf.keras.losses.CategoricalCrossentropy()
        nsamp = train_generator.samples

        num_nonjoker, num_joker = np.bincount(train_generator.labels)
        focalloss_alpha = 0.25 # float(num_nonjoker)/nsamp
        flocalloss_gamma=2
        # loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=focalloss_alpha,gamma=flocalloss_gamma,from_logits=False)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        model.summary(print_fn=log.info)

        class_weight = None # {0:float(num_joker)/nsamp, 1:float(num_nonjoker)/nsamp} # {0: .5, 1: .5}  # not sure about this weighting. should we weight the nonjoker more heavily to avoid false positive jokers?
        stop_early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        save_checkpoint = ModelCheckpoint(checkpoint_filename_path, save_best_only=True, monitor='val_loss', mode='min')
        # Profile from batches 10 to 15
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tb_profiler_log', profile_batch='10, 15')
        plot_callback = PlotHistoryCallback()
        learning_rate_callback = SGDLearningRateTracker()

        log.info('starting training')
        epochs = 300
        steps_per_epoch = 300 # None # 300 # use to reduced # times batches are sampled per epoch. Normally it would be # samples/batch size, e.g. 100k/64
        log.info(f'Training uses class_weight={class_weight} (nonjoker/joker) with max {epochs} epochs steps_per_epoch={steps_per_epoch} optimizer={optimizer} with focalloss_alpha={focalloss_alpha} flocalloss_gamma={flocalloss_gamma} and train_batch_size={train_batch_size} augmenation_enabled={augmentation_enabled}')
        history = None
        log.info(f'learning rate schedule: {lr_schedule}')
        print_datagen_summary(train_generator)
        print_datagen_summary(valid_generator)
        os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1' # to handle interrupt and save model
        try:
            history = model.fit(train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs, verbose=1,
                                validation_data=valid_generator,
                                validation_steps=None,
                                callbacks=[stop_early, save_checkpoint, plot_callback, learning_rate_callback],
                                max_queue_size=10,
                                use_multiprocessing=False,
                                # shuffle = True, # shuffle not valid for folder data generators
                                workers=8,
                                # class_weight=class_weight  # weight nonjokers more to reduce false positives where nonjokers are detected as jokers poking the finger out
                                # it is better to avoid poking out finger until a joker is definitely detected
                                )
            plot_history(history, start_timestr,model_folder)
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
    log.info('making test generator')
    if augmentation_enabled:
        test_datagen = ImageDataGenerator(  # 实例化
            rescale=1. / 255,  # we don't rescale, just leave 0-255
            rotation_range=15,  # 图片随机转动的角度
            width_shift_range=0.2,  # 图片水平偏移的幅度
            height_shift_range=0.2,  # don't shift too much vertically to avoid losing top of card
            fill_mode='constant',
            cval=0,  # fill edge pixels with black; default fills with long lines of color
            zoom_range=[.9, 1.25],  # NOTE zoom >1 minifies, don't zoom in (<1) too much in to avoid losing joker part of card
            # horizontal_flip=False,
        )  # 随机放大或缩小
    else:
        # train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

    log.info('making test generator')
    test_generator = test_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/test/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        # save_to_dir='/tmp/augmented_images',save_prefix='aug', # creates zillions of samples, watch out! make the folder before running or it will not work
        shuffle=False,
        interpolation='nearest',
    )


    gen = test_generator
    print_datagen_summary(gen)
    # gen.reset()
    # loss, acc = model.evaluate(gen, verbose=1)
    # log.info(f'On test set {gen.directory}  loss={loss:.3f}, acc={acc:.4f}')
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
        measure_flops(model)
    except Exception as e:
        log.error(f'Caught {e} measuring flops')
    measure_latency(model_folder)

    elapsed_time_min = (time.time() - start_time) / 60
    if args.train:
        log.info(f'**** done training after {elapsed_time_min:4.1f}m; model saved in {model_folder}.'
                 f'\nSee {LOG_FILE} for logging output for this run.')


def plot_history(history, start_timestr, model_folder=LOG_DIR):
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
            plt.savefig(os.path.join(model_folder, 'accuracy' + '-' + start_timestr + '.png'))
            plt.show()
            # summarize history for loss
            plt.figure('loss')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig(os.path.join(model_folder, 'loss' + '-' + start_timestr + '.png'))
            plt.show()
        except KeyError as k:
            log.warning('could not plot, caught {k}, history.history.keys()={history.history.keys()} ')


def generate_augmented_images(args):
    """
    Generates augmented versions of images in a particular folder and saves them to a new folder.
    Each augmented image has prefix from original PNG image name.
    A dialog selects the source folder and another asks augmentation factor.

    """
    start_timestr = time.strftime("%Y%m%d-%H%M")
    src_data_folder = SRC_DATA_FOLDER
    os.chdir(src_data_folder)
    root = Tk()
    root.withdraw()
    src_data_folder = filedialog.askdirectory()
    if len(src_data_folder) == 0:
        log.info('aborted')
        quit(1)

    train_datagen = ImageDataGenerator(  # 实例化
        #rescale=1. / 255,  # todo check this
        rotation_range=15,  # 图片随机转动的角度
        width_shift_range=0.2,  # 图片水平偏移的幅度
        height_shift_range=0.2,  # don't shift too much vertically to avoid losing top of card
        fill_mode='constant',
        cval=0,  # fill edge pixels with black; default fills with long lines of color
        zoom_range=[.9, 1.25],  # NOTE zoom >1 minifies, don't zoom in (<1) too much in to avoid losing joker part of card
        # horizontal_flip=False,
    )  # 随机放大或缩小

    log.info(f'train_datagen: {train_datagen}')

    aug_factor = simpledialog.askinteger('Augmentation factor','Factor by which to augment?')
    if aug_factor is None:
        log.info('aborted')
        quit(0)

    base=os.path.split(src_data_folder)[-1]
    aug_folder = os.path.join(SRC_DATA_FOLDER,f'{base}-augmented-{start_timestr}')
    data_path = os.path.join(src_data_folder, '*.png')
    files = glob.glob(data_path) # get list of all PNGs
    log.info(f'reading {len(files)} to memory')

    nsrc=len(files)
    ntotal=int(aug_factor*nsrc)
    log.info(f'saving {aug_factor}X={ntotal} augmented images to {aug_folder} from {src_data_folder} with {nsrc} samples')
    try:
        Path(aug_folder).mkdir(parents=True)
    except:
        log.error(f'{aug_folder} already exists')
        quit(1)

    i = 0
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2_dim = 300
    cv2.resizeWindow('original', cv2_dim, cv2_dim)
    for i in range(aug_factor):
        cv2.namedWindow(f'aug {i}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'aug {i}', cv2_dim, cv2_dim)
        cv2.moveWindow(f'aug {i}', (i%5) * cv2_dim, (i//5) * (1+cv2_dim))
    nsaved=0
    ninput=0
    for f in tqdm(files):
        showit=ninput%100==0
        ninput+=1
        img = cv2.imread(f)
        x = img_to_array(img)
        if showit:
            cv2.imshow('original', (1/255.)*x)
        x = x.reshape((1,) + x.shape)
        i=0
        base = os.path.splitext(os.path.split(f)[-1])[0] # e.g. /a/b/c/d/0900111.png gives 0900111
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=aug_folder, save_prefix=base, save_format='png'):
            i += 1
            nsaved+=1
            img_aug=batch[0]
            if showit:
                cv2.imshow(f'aug {i-1}', (1/255.)*img_aug)
                k = cv2.waitKey(15) & 0xff
                if k == ord('x') or k == ord('q'):
                    log.info(f'saved {nsaved} augmented images to {aug_folder} from {src_data_folder}')
                    quit(0)
            if i >= aug_factor:
                break

    log.info(f'saved {nsaved} augmented images to {aug_folder} from {src_data_folder}')

def visualize_model(model=None):
    """ Plots some kernels to visualize their state

    :param model: if supplied, use it, otherwise load latest model
    """
    from matplotlib import pyplot
    from engineering_notation import EngNumber as eng
    model_folder='.'
    if model is None:
        model,interpreter, input_details, output_details,model_folder=load_latest_model(dialog=False,use_tflite_model=False)
    # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    # retrieve weights from the hidden layer
    sel_layer=0
    filters, biases = model.layers[sel_layer].get_weights() # filters is 4d array with [s,s,n,m] with s kernel size, n input, m output, biases is 1d array
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = float(filters.min()), float(filters.max())
    bn=os.path.split(model_folder)[-1]
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = filters.shape[3], 1 # filters.shape[3] total
    ncolrow=math.ceil(math.sqrt(filters.shape[2]*filters.shape[3]))
    fig, axs = pyplot.subplots(ncolrow, ncolrow)
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately

        for j in range(filters.shape[2]): #  total, too many
            # specify subplot and turn of axis
            # ax = pyplot.subplot(n_filters, filters.shape[2], ix)
            ax = pyplot.subplot(ncolrow, ncolrow, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    fig.suptitle(f'{bn} layer {sel_layer}: min/max={eng(f_min)}/{eng(f_max)}')
    pyplot.savefig(os.path.join(model_folder,'kernel_weights.pdf'))
    pyplot.show()



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
    parser.add_argument("--visualize_model", action='store_true', help="plot some kernels from network.")

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
    elif args.visualize_model:
        visualize_model()
    else:
        parser.print_help()


