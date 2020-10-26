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


def rename_imgs(folder):

    dir=folder
    os.chdir(dir)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    ls=os.listdir()
    ls=sorted(ls)
    i=0
    for f in ls:
       if 'png' in f:
            fn=f'tmp/{i:05d}.png'
            i=i+1
            os.rename(f,fn)
    os.chdir('tmp')
    ls=os.listdir()
    for f in ls:
        os.rename(f,'../'+f)
    os.remove('tmp')

def make_train_valid_test():
    SRC_DATA_FOLDER ='/home/tobi/Downloads/trixsyDataset/source_data'
    if not os.path.isdir(TRAIN_DATA_FOLDER):
        log.warning(f'{TRAIN_DATA_FOLDER} does not exist, creating it')
        Path(TRAIN_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    log.info(f'Using source images from {SRC_DATA_FOLDER}')
    os.chdir(SRC_DATA_FOLDER)
    fracs=[.8,.1,.1]
    names=['train','valid','test']

    for i in [1,2]:
        sfn=f'class{i}'
        ls=os.listdir(sfn)
        nfiles=len(ls)-1
        random.shuffle(ls)
        flast=0
        ranges=[]
        for f in fracs:
            ranges.append([math.floor(flast*nfiles), math.floor((flast+f)*nfiles)])
            flast+=f
        filenum=0
        for (n,r) in zip(names,ranges):
            dfn=n+'/'+sfn
            if not os.path.isdir(dfn):
                log.info(f'Creating folder {dfn}')
            Path(dfn).mkdir(parents=True, exist_ok=True)
            for j in tqdm(range(r[0],r[1])):
                sf=sfn+'/'+ls[j]
                if 'png' in sf:
                    nm = f'{filenum:05d}.png'
                elif 'jpg' in sf:
                    nm = f'{filenum:05d}.jpg'
                else:
                    print(f'not image file {sf}')
                    continue

                df=dfn+'/'+nm
                filenum+=1
                copyfile(sf,df)

def test_samples():
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
        log.error('no model loaded')

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
                file_path=class_folder_name + '/' + f
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


# make_train_valid_test()
# test_samples()
