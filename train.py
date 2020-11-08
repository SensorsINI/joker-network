# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train/ valid/ test/ folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
# author: Tobi Delbruck
import argparse
import glob
from pathlib import Path
from random import random
from shutil import copyfile
from tkinter import filedialog
from tkinter import *

# uncomment lines to run on CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# from alessandro: use keras from tensorflow, not from keras directly
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from engineering_notation import EngNumber  as eng # only from pip

from globals_and_utils import *
import datetime

INITIALIZE_MODEL_FROM_LATEST=True # set True to initialize weights to latest saved model

import logging
import sys
log= my_logger(__name__)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def rename_images():
    """ Cleans up a folder filled with images (png and jpg) so that the images are numbered consecutively. Useful after using mv --backup=t to add new images to a folder
    :param folder: the folder name to clean up, relative to working directory
    """
    folder=SRC_DATA_FOLDER
    root = Tk()
    root.withdraw()
    os.chdir(folder)
    folder = filedialog.askdirectory()
    if len(folder)==0:
        log.info('aborted')
        quit(1)
    os.chdir(folder)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    ls = os.listdir() # glob.glob('./*')
    ls=sorted(ls, key=lambda f: os.stat(f).st_mtime)
    log.info(f'folder {folder} has {len(ls)} files')
    i = 0
    log.info('renaming files to tmp folder')
    for f in tqdm(ls):
        if 'png' in f:
            fn = f'tmp/{i:05d}.png'
            i = i + 1
            os.rename(f, fn)
        elif 'jpg' in f:
            fn = f'tmp/{i:05d}.jpg'
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
    NUM_FRAMES_PER_SEGMENT=100 # each 'sample' comes from a consecutive sequence of this many frames to try to avoid that valid/test set have frames that are next to training set frames
    NUM_FRAMES_GAP_BETWEEN_SEGMENTS=20 # each 'sample' comes from a consecutive sequence of this many frames to try to avoid that valid/test set have frames that are next to training set frames

    log.info(f'making training set from {SRC_DATA_FOLDER} with segments of {NUM_FRAMES_PER_SEGMENT} consecutive images with gaps of {NUM_FRAMES_GAP_BETWEEN_SEGMENTS} gaps between segments')
    if not os.path.isdir(TRAIN_DATA_FOLDER):
        log.warning(f'{TRAIN_DATA_FOLDER} does not exist, creating it')
        Path(TRAIN_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    else:
        timestr = time.strftime("%Y%m%d-%H%M")
        backup_folder = f'{TRAIN_DATA_FOLDER}_{timestr}'
        log.warning(f'Renaming existing training folder {TRAIN_DATA_FOLDER} to {backup_folder}')
        os.rename(TRAIN_DATA_FOLDER, backup_folder)

    log.info(f'Using source images from {SRC_DATA_FOLDER}')
    os.chdir(SRC_DATA_FOLDER)
    splits={'train':.8, 'valid':.1, 'test':.1}
    for cls in ['class1', 'class2']:
        ls = os.listdir(cls)
        nfiles = len(ls) - 1
        nsegments=nfiles//(NUM_FRAMES_PER_SEGMENT+NUM_FRAMES_GAP_BETWEEN_SEGMENTS)
        segs=list(range(nsegments))
        random.shuffle(segs)
        log.info(f'{cls} has {nfiles} samples that are split to {nsegments} sequences of {NUM_FRAMES_PER_SEGMENT} frames/seq')
        split_start=0
        for split_name, split_frac in zip(splits.keys(),splits.values()):
            dest_folder_name = os.path.join(TRAIN_DATA_FOLDER, split_name, cls)
            log.info(f'making {dest_folder_name}/ folder for shuffled segments of {NUM_FRAMES_PER_SEGMENT} frames per segment for {split_frac*100:.1f}% {split_name} split of {cls} ')
            Path(dest_folder_name).mkdir(parents=True, exist_ok=True)
            split_end=split_start+split_frac
            seg_range=range(math.floor(split_start*nsegments),math.floor(split_end*nsegments))
            file_nums=[]
            for s in seg_range:
                start=segs[s]*(NUM_FRAMES_PER_SEGMENT+NUM_FRAMES_GAP_BETWEEN_SEGMENTS)
                end=start+NUM_FRAMES_PER_SEGMENT
                nums=list(range(start,end))
                file_nums.extend(nums)
            files=[ls[i] for i in file_nums]
            for file_name in tqdm(files, desc=f'{cls}/{split_name}'):
                source_file_path = os.path.join(SRC_DATA_FOLDER,cls, file_name)
                dest_file_path = os.path.join(dest_folder_name, file_name)
                # print(f'copying {source_file_path} -> {dest_file_path}')
                copyfile(source_file_path, dest_file_path)
            split_start=split_end
    log.info(f'done generating training set from')

def riffle_test():
    """ Runs test on folder of video sequence and pause at detected jokers
    """

    class GetOutOfLoop(Exception):
        pass

    log.info('evaluating riffle')
    log.info(f'Tensorflow version {tf.version.VERSION}')
    interpreter, input_details, output_details=load_tflite_model()
    folder=SRC_DATA_FOLDER
    os.chdir(folder)
    while True:
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        if len(folder)==0:
            log.info('aborted')
            quit(1)
        os.chdir(folder)
        ls=os.listdir()
        ls=sorted(ls)
        first=True
        try:
            while True:
                if first:
                    start=random.randint(0,len(ls))
                    first=False
                else:
                    start=0
                for image_file_path in ls[start:]:
                    if os.path.isdir(image_file_path):
                        continue
                    img = tf.keras.preprocessing.image.load_img(image_file_path,color_mode='grayscale')
                    input_arr = tf.keras.preprocessing.image.img_to_array(img)
                    img =np.array(input_arr)
                    dec, joker_prob, pred=classify_joker_img(img, interpreter, input_details, output_details)
                    if dec==1:
                        cv2.putText(img,'Joker',(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                        print('\a') # beep on some terminals https://stackoverflow.com/questions/6537481/python-making-a-beep-noise
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.imshow('frame', np.array(img))
                    cv2.resizeWindow('frame', 800, 400)

                    if dec == 0:
                        k = cv2.waitKey(15) & 0xff
                    else:
                        k = cv2.waitKey(2000) & 0xff
                    if k == 27 or k == ord('q') or k == ord('x'):
                        cv2.destroyAllWindows()
                        quit()
                    elif k!=255:
                        os.chdir('..')
                        raise GetOutOfLoop # choose new folder
        except GetOutOfLoop:
            continue


def test_random_samples():
    """ Runs test on test folder to evaluate accuracy on example images
    """
    import glob
    import tensorflow as tf
    from tensorflow.python.keras.models import load_model
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
    idx = [0,0]
    for c in [1, 2]:
        class_folder_name.append(test_folder + f'class{c}')
        ls.append(os.listdir(class_folder_name[c-1]))
        random.shuffle(ls[c-1])
    while True:
        windows = []
        for c in [1, 2]:
            gt_class = 'nonjoker' if c == 1 else 'joker'
            image_file_name=ls[c - 1][idx[c - 1]]
            image_file_path = os.path.join(class_folder_name[c-1], image_file_name)
            img = tf.keras.preprocessing.image.load_img(image_file_path,color_mode='grayscale')
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = (1./255)*np.array([input_arr])  # Convert single image to a batch.
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
            idx[c-1] += 1
            if idx[c-1] >= len(ls[c-1]): idx[c-1] = 0


def classify_joker_img(img: np.array, interpreter, input_details, output_details):
    """ Classify uint8 img

    :param img: input image as unit8 np.array
    :param interpreter: the TFLITE interpreter
    :param input_details: the input details of interpreter
    :param output_details: the output details of interpreter

    :returns: decision (0 or 1), joker_probability (0-1), prediction[2]=[nonjoker, joker]
    """
    interpreter.set_tensor(input_details[0]['index'], (1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    dec = np.argmax(pred[0])
    joker_prob = pred[0][1]
    return dec, joker_prob, pred


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


def load_tflite_model():
    """ loads the most recent trained TFLITE model

    :returns: interpreter,input_details,output_details
    """
    existing_models = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '_*/')
    tflite_model_path = None
    if len(existing_models) > 0:
        latest_model_folder = max(existing_models, key=os.path.getmtime)
        tflite_model_path = os.path.join(latest_model_folder, TFLITE_FILE_NAME)
        if not os.path.isfile(tflite_model_path):
            log.error(f'no TFLITE model found at {tflite_model_path}')
            quit(1)
    else:
        log.error(f'no models found in {MODEL_DIR}')
        quit(1)
    log.info('loading latest tflite CNN model {}'.format(tflite_model_path))
    # model = load_model(MODEL)
    # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def get_flops():
    log.info('measuring Op/frame for CNN')
    from flopco_keras import FlopCoKeras
    import tensorflow as tf

    model = create_model() # load_latest_model() #tf.keras.applications.ResNet101()
    flopco=FlopCoKeras(model)
    flopco.get_stats()

    # log.info(f'flop counter: {str(flopco)}')
    log.info(f"Op/frame: {eng(flopco.total_flops)}")
    log.info(f"MAC/frame: {eng(flopco.total_macs)}")
    s='Fractional Op per layer: '
    for f in flopco.relative_flops:
        s=s+f' {f*100:.2f}%'
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

def create_model():
    """ Creates the CNN model for joker detection
    """
    model = Sequential()

    # model.add(Input(shape=(None,IMSIZE,IMSIZE,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=64, kernel_size=(11, 11),
                     strides=(4, 4), padding='valid',
                     input_shape=(IMSIZE, IMSIZE, 1),
                     activation='relu', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                     strides=(1, 1), padding='same',
                     activation='relu', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
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
    # model.add(Dense(4096, activation='relu', name='fc6'))
    # model.add(Dropout(0.5))

    # model.add(Dense(4096, activation='relu', name='fc7'))
    # model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu', name='fc8'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(2, activation='softmax', name='output'))

    return model


def measure_latency():
    log.info('measuring CNN latency in loop')
    interpreter,input_details,output_details=load_tflite_model()
    img=np.random.randint(0,255,(IMSIZE,IMSIZE,1))
    N=100
    for i in range(1,N):
        with Timer('CNN latency') as timer:
            classify_joker_img(img,interpreter,input_details,output_details)
    timer.print_timing_info(log)


class PlotHistoryCallback(History):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch,logs)
        keys = list(logs.keys())
        log.info("End epoch {} of training; got log keys: {}".format(epoch, keys))
        plot_history(self.model.history, 'history')


def train(args=None):
    model_name = None
    tflite_model_name = None

    start_time = time.time()
    start_timestr = time.strftime("%Y%m%d-%H%M")
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f'training-{start_timestr}.log')
    fh = logging.FileHandler(LOG_FILE, 'w')  # 'w' to overwrite, not append
    fh.setLevel(logging.INFO)
    fmtter = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(message)s")
    fh.setFormatter(fmtter)
    log.addHandler(fh)

    log.info(f'Tensorflow version {tf.version.VERSION}')
    log.info(f'dataset path: TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}')
    log.info(f'TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}\nSRC_DATA_FOLDER={SRC_DATA_FOLDER}')

    num_classes = 2
    checkpoint_filename_path = os.path.join(MODEL_DIR, 'models/joker_net_checkpoint.hdf5')

    model = create_model()
    latest_existing_model_folder=None
    if INITIALIZE_MODEL_FROM_LATEST:
        existing_model_folders = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '*/')
        TIMEOUT = 30
        if len(existing_model_folders) > 0:
            latest_existing_model_folder = max(existing_model_folders, key=os.path.getmtime)
            getmtime_checkpoint = os.path.getmtime(checkpoint_filename_path) if os.path.isfile(checkpoint_filename_path) else 0
            getmtime_stored_model = os.path.getmtime(latest_existing_model_folder)
            if os.path.isfile(checkpoint_filename_path) \
                    and getmtime_checkpoint > getmtime_stored_model \
                    and yes_or_no(f'checkpoint {checkpoint_filename_path} modified {datetime.datetime.fromtimestamp(getmtime_checkpoint)} \nis newer than saved model {existing_model_folders[0]} modified {datetime.datetime.fromtimestamp(getmtime_stored_model)},\n start from it?', timeout=TIMEOUT):
                log.info(f'loading weights from checkpoint {checkpoint_filename_path}')
                model.load_weights(checkpoint_filename_path)
            else:
                if yes_or_no(f'model {latest_existing_model_folder} exists, start from it?', timeout=TIMEOUT):
                    log.info(f'initializing model from {latest_existing_model_folder}')
                    model = load_model(latest_existing_model_folder)
                else:
                    log.info('creating new empty model')
                    model = create_model()
        else:
            if os.path.isfile(checkpoint_filename_path) and yes_or_no('checkpoint exists, start from it?', timeout=TIMEOUT):
                log.info(f'loading weights from checkpoint {checkpoint_filename_path}')
                model = create_model()
                model.load_weights(checkpoint_filename_path)
            else:
                yn = yes_or_no("Could not find saved model or checkpoint. Initialize a new model?", timeout=TIMEOUT)
                if yn:
                    log.info('creating new empty model')
                    model = create_model()
                else:
                    log.warning('aborting training')
                    quit(1)
    else:
        log.info('creating new empty model')
    optimizer=tf.keras.optimizers.SGD(momentum=.9) # alessandro: SGD gives higher accuracy than Adam but include a momentuum
    loss=tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary(print_fn=log.info)

    train_batch_size = 128
    valid_batch_size = 64
    test_batch_size = 64

    train_datagen = ImageDataGenerator(  # 实例化
        rescale=1. / 255,  # todo check this
        rotation_range=20,# 图片随机转动的角度
        width_shift_range=0.25,  # 图片水平偏移的幅度
        height_shift_range=0.25,  # 图片竖直偏移的幅度
        fill_mode='constant',
        cval=0, # fill edge pixels with black
        # zoom_range=0.2,
        # horizontal_flip=False,
    )  # 随机放大或缩小

    log.info('making training generator')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/train/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        # save_to_dir='/tmp/augmented_images',save_prefix='aug', # creates zillions of samples, watch out! make the folder before running or it will not work
        shuffle=True,
    )

    log.info('making validation generator')
    valid_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/valid/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
    )

    log.info('making test generator')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(  # 实例化
        TRAIN_DATA_FOLDER + '/test/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
    )  # IMPORTANT shuffle=False here or model.predict will NOT match GT of test generator in test_generator.labels!

    # Path('test_gen_samples').mkdir(parents=True, exist_ok=True)

    def print_datagen_summary(gen: ImageDataGenerator):
        nsamp = gen.samples
        num_nonjoker, num_joker = np.bincount(gen.labels)

        def pc(n):
            return 100 * float(n) / nsamp

        log.info(f'summary of {gen.directory}:'
                 f' {gen.samples} samples:\t{num_nonjoker}/{pc(num_nonjoker):.3f}% nonjoker,\t{num_joker}/{pc(num_joker):.3f}% joker\n')

    print_datagen_summary(train_generator)
    print_datagen_summary(valid_generator)
    print_datagen_summary(test_generator)

    stop_early = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
    save_checkpoint = ModelCheckpoint(checkpoint_filename_path, save_best_only=True, monitor='val_loss', mode='min')
    # Profile from batches 10 to 15
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tb_profiler_log', profile_batch='10, 15')
    plot_callback=PlotHistoryCallback()


    if args.train:
        log.info('starting training')
        epochs=300

        class_weight = {0: .7, 1: .3}
        log.info(f'Training uses class_weight={class_weight} (nonjoker/joker) with max {epochs} epochs optimizer={optimizer} and train_batch_size={train_batch_size} ')
        history = None


        try:
            history = model.fit(train_generator,  # todo back to train generator
                                # steps_per_epoch=150,
                                epochs=epochs, verbose=1,
                                validation_data=valid_generator,
                                # validation_steps=25,
                                callbacks=[stop_early, save_checkpoint, plot_callback],
                                max_queue_size=20,
                                use_multiprocessing=False,
                                # shuffle = True, # shuffle not valid for folder data generators
                                workers=8,
                                class_weight=class_weight  # weight nonjokers more to reduce false positives where nonjokers are detected as jokers poking the finger out
                                # it is better to avoid poking out finr until a joker is definitely detected
                                )
            plot_history(history, start_timestr)
            training_history_filename = os.path.join(LOG_DIR, 'training_history' + '-' + start_timestr + '.npy')
            np.save(training_history_filename, history.history)
            log.info(f'Done with model.fit; history is \n{history.history} and is saved as {training_history_filename}')
            log.info(f'history.history.keys()={history.history.keys()}')

        except KeyboardInterrupt:
            log.warning('keyboard interrupt, saving model and testing')

        new_model_folder_name = os.path.join(MODEL_DIR, f'{JOKER_NET_BASE_NAME}_{start_timestr}')

        log.info(f'saving model to folder {new_model_folder_name}')
        model.save(new_model_folder_name)

        log.info('converting model to tensorflow lite model')
        converter = tf.lite.TFLiteConverter.from_saved_model(new_model_folder_name)  # path to the SavedModel directory
        tflite_model = converter.convert()
        tflite_model_path = os.path.join(new_model_folder_name, TFLITE_FILE_NAME)

        log.info(f'saving tflite model as {tflite_model_path}')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        # end training part

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

    get_flops()
    measure_latency()
    elapsed_time_min = (time.time() - start_time) / 60
    if args.train:
        log.info(f'**** done training after {elapsed_time_min:4.1f}m; model saved in {new_model_folder_name}.'
             f'\nSee {LOG_FILE} for logging output for this run.')


def plot_history(history, start_timestr):
    if history is not None:
        try:

            # summarize history for accuracy
            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig(os.path.join(LOG_DIR, 'accuracy' + '-' + start_timestr + '.png'))
            plt.show()
            # summarize history for loss
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


args=None # if run as module

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train:train CNN for trixsy', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", action='store_true', help="train model starting from latest (shows options to choose whether to initialize.")
    parser.add_argument("--test_accuracy", action='store_true', help="run network test rather than train.")
    parser.add_argument("--riffle_test", action='store_true', help="test by pausing playback/classify of recorded frames at detected jokers.")
    parser.add_argument("--rename_images",  action='store_true', help="rename images in a folder consecutively sorting them by mtime.")
    parser.add_argument("--make_training_set", action='store_true', help="make training data from source images.")
    parser.add_argument("--test_random_samples", action='store_true', help="test random samples from test set.")
    parser.add_argument("--measure_flops", action='store_true', help="measures flops/frame of network.")
    parser.add_argument("--measure_latency", action='store_true', help="measures CNN latency.")

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
        riffle_test()
    elif args.measure_latency:
        measure_latency()
    elif args.measure_flops:
        stats=get_flops()
        log.info(f'total flops/frame={stats.total_flops}')
    else:
        parser.print_help()

