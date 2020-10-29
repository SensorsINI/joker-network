# renumber all files in a folder
import math
import os
from shutil import copyfile
import random
# import itertools
from pathlib import Path
from globals_and_utils import *
from tqdm import tqdm
log=my_logger(__name__)


def rename_images(folder):
    """ Cleans up a folder filled with images (png and jpg) so that the images are numbered consecutively. Useful after using mv --backup=t to add new images to a folder
    :param folder: the folder name to clean up, relative to working directory
    """
    os.chdir(folder)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    ls=os.listdir()
    log.info(f'folder {folder} has {len(ls)} files')
    ls=sorted(ls)
    i=0
    log.info('renaming files to tmp folder')
    for f in tqdm(ls):
       if 'png' in f:
            fn=f'tmp/{i:05d}.png'
            i=i+1
            os.rename(f,fn)
       elif 'jpg' in f:
           fn = f'tmp/{i:05d}.jpg'
           i = i + 1
           os.rename(f, fn)

    os.chdir('tmp')
    ls=os.listdir()
    log.info('moving files back to src folder')
    for f in tqdm(ls):
        os.rename(f,'../'+f)
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
        os.rename(TRAIN_DATA_FOLDER,backup_folder)

    log.info(f'Using source images from {SRC_DATA_FOLDER}')
    os.chdir(SRC_DATA_FOLDER)
    SPLIT=[.8,.1,.1]
    names=['train','valid','test']

    for i in [1,2]:
        sfn=f'class{i}'
        ls=os.listdir(sfn)
        nfiles=len(ls)-1
        random.shuffle(ls)
        flast=0
        ranges=[]
        for f in SPLIT:
            ranges.append([math.floor(flast*nfiles), math.floor((flast+f)*nfiles)])
            flast+=f
        for (n,r) in zip(names,ranges):
            dfn=os.path.join(TRAIN_DATA_FOLDER,n,sfn)
            log.info(f'making {dfn}/ folder for shuffled files from {sfn} in range [{r[0]},{r[1]}')
            Path(dfn).mkdir(parents=True, exist_ok=True)
            for j in tqdm(range(r[0],r[1]),desc=f'{sfn}/{n}'):
                sf=os.path.join(sfn,ls[j])
                df=os.path.join(dfn,ls[j])
                # if j-r[0]<3 or j==r[1]-1:
                #     print(f'copying {sf} -> {df}')
                # elif j-r[0]==3:
                #     print('...')
                copyfile(sf,df)

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
    model=None
    if len(existing_model_folders) > 0:
        latest_model_folder = max(existing_model_folders, key=os.path.getmtime)
        log.info(f'initializing model from {latest_model_folder}')
        model = load_model(latest_model_folder)
    else:
        log.error('no model found to load')
        quit(1)

    NUM_SAMPLES=1
    test_folder=TRAIN_DATA_FOLDER + '/test/'
    import random
    while True:
        windows=[]
        for c in [1,2]:
            gt_class='nonjoker' if c==1 else 'joker'
            class_folder_name=test_folder+f'class{c}'
            ls=os.listdir(class_folder_name)
            random.shuffle(ls)
            for f in ls[0:NUM_SAMPLES]:
                file_path=os.path.join(class_folder_name,f)
                img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(IMSIZE,IMSIZE))
                input=(1. / 255) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]))
                pred = model.predict(input)
                dec = 'joker' if np.argmax(pred[0])==1 else 'nonjoker'
                correct='right' if ((dec=='joker' and c==2) or (dec=='nonjoker' and c==1)) else 'wrong'
                if correct=='wrong': # save wrong classifications for later
                    copy_folder=TRAIN_DATA_FOLDER+'/incorrect/'+f'class{c}'
                    Path(copy_folder).mkdir(parents=True, exist_ok=True)
                    log.info(f'saving file {f} as incorrect {gt_class} classified as {dec}')
                    copyfile(file_path, os.path.join(copy_folder,f))
                joker_prob = pred[0][1]
                win_name=f'{correct}: class{c}: {dec} (joker_prob={joker_prob:.2f})'
                # cv2.namedWindow(win_name)
                windows.append(win_name)
                cv2.imshow(win_name,img)
                cv2.moveWindow(win_name,1,500*(c-1)+1)
                k = cv2.waitKey(100) & 0xff
                if correct=='wrong':
                    time.sleep(3)
        if k==27 or k==ord('q'):
            break
        for w in windows:
            cv2.destroyWindow(w)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset utilities', allow_abbrev=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rename_images", type=str, default=None, help="rename images in the folder consecutively.")
    parser.add_argument("--make_training_set", action='store_true', help="make training data from source images.")
    parser.add_argument("--test_random_samples", action='store_true', help="test random samples from test set.")

    print(f'TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}\nSRC_DATA_FOLDER={SRC_DATA_FOLDER}')
    args=parser.parse_args()
    if args.rename_images:
        rename_images(args.rename_images)
    elif args.make_training_set:
        make_training_set()
    elif args.test_random_samples:
        test_random_samples()
    else:
        parser.print_help()




