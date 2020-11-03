# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train/ valid/ test/ folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
# author: Tobi Delbruck
import argparse
import glob
from pathlib import Path
from random import random

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from globals_and_utils import *
import datetime

INITIALIZE_MODEL_FROM_LATEST=True # set True to initialize weights to latest saved model

log= my_logger(__name__)
log.setLevel(LOGGING_LEVEL)

def rename_images(folder):
    """ Cleans up a folder filled with images (png and jpg) so that the images are numbered consecutively. Useful after using mv --backup=t to add new images to a folder
    :param folder: the folder name to clean up, relative to working directory
    """
    os.chdir(folder)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    ls = os.listdir()
    log.info(f'folder {folder} has {len(ls)} files')
    ls = sorted(ls)
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
    SPLIT = [.8, .1, .1]
    names = ['train', 'valid', 'test']

    for i in [1, 2]:
        sfn = f'class{i}'
        ls = os.listdir(sfn)
        nfiles = len(ls) - 1
        random.shuffle(ls)
        flast = 0
        ranges = []
        for f in SPLIT:
            ranges.append([math.floor(flast * nfiles), math.floor((flast + f) * nfiles)])
            flast += f
        for (n, r) in zip(names, ranges):
            dfn = os.path.join(TRAIN_DATA_FOLDER, n, sfn)
            log.info(f'making {dfn}/ folder for shuffled files from {sfn} in range [{r[0]},{r[1]}')
            Path(dfn).mkdir(parents=True, exist_ok=True)
            for j in tqdm(range(r[0], r[1]), desc=f'{sfn}/{n}'):
                sf = os.path.join(sfn, ls[j])
                df = os.path.join(dfn, ls[j])
                # if j-r[0]<3 or j==r[1]-1:
                #     print(f'copying {sf} -> {df}')
                # elif j-r[0]==3:
                #     print('...')
                copyfile(sf, df)


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
            correct = 'right' if ((dec == 'joker' and c == 2) or (dec == 'nonjoker' and c == 1)) else 'wrong'
            if correct == 'wrong':  # save wrong classifications for later
                copy_folder = TRAIN_DATA_FOLDER + '/incorrect/' + f'class{c}'
                Path(copy_folder).mkdir(parents=True, exist_ok=True)
                log.info(f'saving file {image_file_name} as incorrect {gt_class} classified as {dec}')
                copyfile(image_file_path, os.path.join(copy_folder, image_file_name))
            joker_prob = pred[0][1]
            win_name = f'{correct}: Real class:class{c}/{gt_class} detected as {dec} (joker_prob={joker_prob:.2f})'
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


def get_flops():
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(MODEL)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            flops_log_path = os.path.join('.', 'tf_flops_log.txt')
            opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops

# AlexNet
def create_model():
    model = Sequential()

    # model.add(Input(shape=(None,IMSIZE,IMSIZE,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=64, kernel_size=(11,11),
                     strides=(4,4), padding='valid',
                     input_shape=(IMSIZE,IMSIZE,1),
                     activation='relu', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2),
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(5,5),
                     strides=(1,1), padding='same',
                     activation='relu', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2),
                           padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3,3),
                     strides=(1,1), padding='same',
                     activation='relu', name='conv3'))
    model.add(Conv2D(filters=128, kernel_size=(3,3),
                     strides=(1,1), padding='same',
                     activation='relu', name='conv4'))
    #model.add(Conv2D(filters=256, kernel_size=(3,3),
    #                 strides=(1,1), padding='same',
    #                 activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2), padding='valid'))

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


def train():
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
    checkpoint_filename = 'joker_net_checkpoint.hdf5'

    model = create_model()
    if INITIALIZE_MODEL_FROM_LATEST:
        existing_model_folders = glob.glob(MODEL_DIR + '/' + JOKER_NET_BASE_NAME + '*/')
        TIMEOUT = 30
        if len(existing_model_folders) > 0:
            latest_model_folder = max(existing_model_folders, key=os.path.getmtime)
            getmtime_checkpoint = os.path.getmtime(checkpoint_filename) if os.path.isfile(checkpoint_filename) else 0
            getmtime_stored_model = os.path.getmtime(latest_model_folder)
            if os.path.isfile(checkpoint_filename) \
                    and getmtime_checkpoint > getmtime_stored_model \
                    and yes_or_no(f'checkpoint {checkpoint_filename} modified {datetime.datetime.fromtimestamp(getmtime_checkpoint)} \nis newer than saved model {existing_model_folders[0]} modified {datetime.datetime.fromtimestamp(getmtime_stored_model)},\n start from it?', timeout=TIMEOUT):
                log.info(f'loading weights from checkpoint {checkpoint_filename}')
                model.load_weights(checkpoint_filename)
            else:
                if yes_or_no(f'model {latest_model_folder} exists, start from it?', timeout=TIMEOUT):
                    log.info(f'initializing model from {latest_model_folder}')
                    model = load_model(latest_model_folder)
                else:
                    log.info('creating new empty model')
                    model = create_model()
        else:
            if os.path.isfile(checkpoint_filename) and yes_or_no('checkpoint exists, start from it?', timeout=TIMEOUT):
                log.info(f'loading weights from checkpoint {checkpoint_filename}')
                model = create_model()
                model.load_weights(checkpoint_filename)
            else:
                yn = yes_or_no("Could not find saved model or checkpoint. Initialize a new model?", timeout=TIMEOUT)
                log.info('creating new empty model')
                model = create_model()
    else:
        log.info('creating new empty model')
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['categorical_accuracy'])
    model.summary(print_fn=log.info)

    train_batch_size = 64
    valid_batch_size = 64
    test_batch_size = 64

    log.info('making training generator')
    train_datagen = ImageDataGenerator(  # 实例化
        rescale=1. / 255,  # todo check this
        rotation_range=20,  # 图片随机转动的角度
        width_shift_range=0.3,  # 图片水平偏移的幅度
        height_shift_range=0.3,  # 图片竖直偏移的幅度
        zoom_range=0.2,
        horizontal_flip=False)  # 随机放大或缩小

    log.info('making training generator')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/train/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True)

    log.info('making validation generator')
    valid_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/valid/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True)

    log.info('making test generator')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(  # 实例化
        TRAIN_DATA_FOLDER + '/test/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
        # save_to_dir='test_gen_samples',
        # save_prefix=''
    )  # IMPORTANT shuffle=False here or model.predict will NOT match GT of test generator in test_generator.labels!

    # Path('test_gen_samples').mkdir(parents=True, exist_ok=True)

    def print_datagen_summary(gen: ImageDataGenerator):
        nsamp = gen.samples
        num_joker, num_nonjoker = np.bincount(gen.labels)

        def pc(n):
            return 100 * float(n) / nsamp

        log.info(f'summary of {gen.directory}:'
                 f' {gen.samples} samples:\t{pc(num_nonjoker):.1f}% nonjoker,\t{pc(num_joker):.1f}% joker\n')

    print_datagen_summary(train_generator)
    print_datagen_summary(valid_generator)
    print_datagen_summary(test_generator)

    stop_early = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
    save_checkpoint = ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')
    # Profile from batches 10 to 15
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tb_profiler_log',
                                                 profile_batch='10, 15')

    if args is None or not args.test_accuracy:
        log.info('starting training')
        history = None
        try:
            history = model.fit(train_generator,  # todo back to train generator
                                # steps_per_epoch=150,
                                epochs=300, verbose=1,
                                validation_data=valid_generator,
                                # validation_steps=25,
                                callbacks=[stop_early, save_checkpoint, tb_callback],
                                # max_queue_size=capacity,
                                # shuffle = True, # shuffle not valid for folder data generators
                                workers=1,
                                class_weight={0: 5, 1: 1}  # weight nonjokers more to reduce false positives where nonjokers are detected as jokers poking the finger out
                                # it is better to avoid poking out finr until a joker is definitely detected
                                )
            if history is not None:
                try:
                    training_history_filename = 'training_history.npy'
                    np.save(training_history_filename, history.history)
                    log.info(f'Done with model.fit; history is \n{history.history} and is saved as {training_history_filename}')
                    log.info(f'history.history.keys()={history.history.keys()}')

                    # summarize history for accuracy
                    plt.plot(history.history['categorical_accuracy'])
                    plt.plot(history.history['val_categorical_accuracy'])
                    plt.title('model accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'validate'], loc='upper left')
                    plt.show()
                    # summarize history for loss
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'validate'], loc='upper left')
                    plt.show()
                except KeyError as k:
                    log.warning('could not plot, caught {k}, history.history.keys()={history.history.keys()} ')
        except KeyboardInterrupt:
            log.warning('keyboard interrupt, saving model and testing')

        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        model_folder = os.path.join(MODEL_DIR, f'{JOKER_NET_BASE_NAME}_{start_timestr}')

        log.info(f'saving model to folder {model_folder}')
        model.save(model_folder)

        log.info('converting model to tensorflow lite model')
        converter = tf.lite.TFLiteConverter.from_saved_model(model_folder)  # path to the SavedModel directory
        tflite_model = converter.convert()
        tflite_model_path = os.path.join(MODEL_DIR, f'{model_folder}.tflite')

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
    log.info(f'**** final test set balanced accuracy: {balanced_accuracy * 100:6.3f}\% (chance would be 50\%)\nConfusion matrix nonjoker/joker:\n {conf_matrix}')
    np.set_printoptions(precision=2)
    # disp = plot_confusion_matrix(classifier, X_test, y_test,
    #                                  display_labels=['nonjoker','joker'],
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=True)
    # disp.ax_.set_title('joker/nonjoker confusion matrix')

    elapsed_time_min = (time.time() - start_time) / 60
    log.info(f'**** done training after {elapsed_time_min:4.1f}m; model saved in {model_folder} and {tflite_model_path}.'
             f'\nSee {LOG_FILE} for logging output for this run.')


args=None # if run as module
model_folder=None
tflite_model_path=None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train:train CNN for txsy', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_accuracy", action='store_true', help="run network test rather than train")
    parser.add_argument("--rename_images", type=str, default=None, help="rename images in the folder consecutively.")
    parser.add_argument("--make_training_set", action='store_true', help="make training data from source images.")
    parser.add_argument("--test_random_samples", action='store_true', help="test random samples from test set.")
    parser.add_argument("--measure_flops", action='store_true', help="measures flops/frame of network.")

    args = parser.parse_args()
    if args.train:
        train()
    elif args.rename_images:
        rename_images(args.rename_images)
    elif args.make_training_set:
        make_training_set()
    elif args.test_random_samples:
        test_random_samples()
    elif args.measure_flops:
        log.info(f'total flops/frame={get_flops():.3e}')
    else:
        parser.print_help()

