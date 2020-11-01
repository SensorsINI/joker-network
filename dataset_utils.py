# useful utilities for dataset prep and other stuff for Trixsy
# author: Tobi Delbruck

import math
import os
from shutil import copyfile
import random
# import itertools
from pathlib import Path
from globals_and_utils import *
from tqdm import tqdm
import tensorflow as tf

log = my_logger(__name__)


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
    existing_model_folders = glob.glob(JOKER_NET_BASE_NAME + '*/')
    model = None
    # if len(existing_model_folders) > 0:
    #     log.info(f'found existing models:\n{existing_model_folders}\n choosing newest one')
    #     latest_model_folder = max(existing_model_folders, key=os.path.getmtime)
    #     log.info(f'*** initializing model from {latest_model_folder}')
    #     time.sleep(5)
    #     model = load_model(latest_model_folder)
    #     # model.compile()
    #     model.summary()
    #     print(f'model.input_shape: {model.input_shape}')
    #
    #
    # else:
    #     log.error('no model found to load')
    #     quit(1)

    # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert
    existing_models = glob.glob(JOKER_NET_BASE_NAME + '_*.tflite')
    latest_model = max(existing_models, key=os.path.getmtime)
    log.info(f'loading latest tflite model {latest_model}')
    time.sleep(5)
    interpreter = tf.lite.Interpreter(model_path=latest_model)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
            f=ls[c - 1][idx[c - 1]]
            file_path = os.path.join(class_folder_name[c-1], f)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (IMSIZE, IMSIZE))
            # input = (1. / 255) * np.array(np.reshape(img, [IMSIZE, IMSIZE, 1]),dtype=np.float32)
            # pred1 = model.predict(input[None,:])
            # input=(1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]))
            # pred = model.predict(input)
            interpreter.set_tensor(input_details[0]['index'], (1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])

            dec = 'joker' if np.argmax(pred[0]) == 1 else 'nonjoker'
            correct = 'right' if ((dec == 'joker' and c == 2) or (dec == 'nonjoker' and c == 1)) else 'wrong'
            if correct == 'wrong':  # save wrong classifications for later
                copy_folder = TRAIN_DATA_FOLDER + '/incorrect/' + f'class{c}'
                Path(copy_folder).mkdir(parents=True, exist_ok=True)
                log.info(f'saving file {f} as incorrect {gt_class} classified as {dec}')
                copyfile(file_path, os.path.join(copy_folder, f))
            joker_prob = pred[0][1]
            win_name = f'{correct}: Real class:class{c}/{gt_class} detected as {dec} (joker_prob={joker_prob:.2f})'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            windows.append(win_name)
            cv2.imshow(win_name, img)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset utilities', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rename_images", type=str, default=None, help="rename images in the folder consecutively.")
    parser.add_argument("--make_training_set", action='store_true', help="make training data from source images.")
    parser.add_argument("--test_random_samples", action='store_true', help="test random samples from test set.")
    parser.add_argument("--measure_flops", action='store_true', help="measures flops/frame of network.")

    print(f'TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}\nSRC_DATA_FOLDER={SRC_DATA_FOLDER}')
    args = parser.parse_args()
    if args.rename_images:
        rename_images(args.rename_images)
    elif args.make_training_set:
        make_training_set()
    elif args.test_random_samples:
        test_random_samples()
    elif args.measure_flops:
        log.info(f'total flops/frame={get_flops():.3e}')
    else:
        parser.print_help()
