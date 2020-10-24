# renumber all files in a folder
import math
import os
from shutil import copyfile
import random
# import itertools
from pathlib import Path

DS='/home/tobi/Downloads/trixsyDataset/'
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
    os.chdir(DS)
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
            Path(dfn).mkdir(parents=True, exist_ok=True)
            for j in range(r[0],r[1]):
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

make_train_valid_test()

