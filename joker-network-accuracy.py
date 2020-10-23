from __future__ import absolute_import, division, print_function, unicode_literals

import os,sys
import tensorflow as tf
import os, shutil, random, glob
import cv2
import numpy as np
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from globals_and_utils import *


print(tf.version.VERSION)
imgdir = TRAIN_DATA_FOLDER
print('dataset path', imgdir)

resize = IMSIZE

def get_img(img_paths, img_size):
    X = np.zeros((len(img_paths),img_size,img_size,1),dtype=np.uint8)
    i = 0
    for img_path in img_paths:
        img = cv2.imread(img_path, 0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
        X[i,:,:,:] = img
        i += 1
    return X

def get_X_batch(X_path, batch_size, img_size):
    while 1:
        for i in range(0, len(X_path), batch_size):
            X = get_img(X_path[i:i+batch_size], img_size)
            yield X

# images = get_X_batch(img_paths,16,300) #得到一个batch的图片，形式为generator
# images = next(images) #next(generator)，得到一个batch的ndarray
# images_aug = seq.augment_images(images) #得到增强后的图片ndarray


classnum=2
def load_data():
 
    print('---------------', dataset_size)
    train_data = np.empty((dataset_size - testsize, resize, resize, 1), dtype="uint8")
    train_label = np.empty((dataset_size - testsize,), dtype="int32")
    test_data = np.empty((testsize, resize, resize, 1), dtype="uint8")
    test_label = np.empty((testsize, ), dtype="int32")
    for i, img in enumerate(imgs):
        tmp = img.split('.')[0]
        fileid = tmp.split('_')[1]
     
        imgc = cv2.imread(imgdir+img, 0)
        imgf = cv2.resize(imgc, (resize, resize),interpolation=cv2.INTER_AREA)
        imgf = np.reshape(imgf, (resize, resize, 1))
   
        if i < testsize:
          
            test_data[i] = imgf
            test_label[i] = fileid
  
        else:
       
            
            train_data[i-testsize] = imgf
            train_label[i-testsize] = fileid
        
    return train_data, train_label, test_data, test_label


# function for reading images
def get_im_cv2(paths, img_size, color_type=1, normalize=False):
    '''
    paras:
        paths：path list of images
        img_rows:
        img_cols:
        color_type: RGB or GRAY 3 or 1
    return:
        imgs: images array
    '''
    # Load as grayscale
    # imgs = []
    i = 0
    X = np.zeros((len(img_paths),img_size,img_size,color_type),dtype=np.uint8)
    for path in paths:
        
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        # Reduce size
        img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
        if normalize:
            resized = resized.astype('float32')
            resized /= 127.5
            resized -= 1. 
        X[i,:,:,None] = img

        i += 1
        # imgs.append(resized)
        
    # return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)
    return X

def get_train_batch(X_train, y_train, batch_size, img_size, color_type, is_argumentation):
    '''
    para:
        X_train：path list of training images
        y_train: label list of images correspondingly
        batch_size:
        img_w:
        img_h:
        color_type:
        is_argumentation:
    return:
        a generator，x: batch images y: labels
    '''
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = get_im_cv2(X_train[i:i+batch_size], img_size, color_type)
            y = y_train[i:i+batch_size]
            if is_argumentation:
                
                x, y = img_augmentation(x, y)
            # this yield is important, represents return, after retuen the loop is still working, and return.
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield(np.array(x), np.array(y))


#LeNet-5
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(5,5), 
#                  padding='valid', 
#                  input_shape=(184, 184, 3), 
#                  activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=16, kernel_size=(5,5), 
#                  padding='valid', 
#                  activation='tanh')) #
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(120, activation='tanh'))
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(2, activation='softmax')) #softmax函数做激活函数 e^x/(sum(e^x))

# sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=True) #采用随机梯度下降法作为优化算法
# model.compile(loss='binary_crossentropy',
#               optimizer=sgd, 
#               metrics=['accuracy'])

# AlexNet
def create_model():
    model = Sequential()

    # model.add(Input(shape=(None,resize,resize,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=64, kernel_size=(11,11),
                     strides=(4,4), padding='valid',
                     input_shape=(resize,resize,1),
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
    model.add(Dense(classnum, activation='softmax', name='output'))

    return model
# model.add(Activation('softmax'))


# model = create_model()

#model = load_model('alexnet184grayv22.h5')
# model.summary()

model = load_model(sys.argv[2])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# model.summary()

train_batch_size = 128
valid_batch_size = 64
test_batch_size = 32

train_datagen = ImageDataGenerator( #实例化
    rescale=1./255,
    rotation_range = 90,  #图片随机转动的角度
    width_shift_range = 0.2, #图片水平偏移的幅度
    height_shift_range = 0.2, #图片竖直偏移的幅度
    zoom_range = 0.2,
    horizontal_flip=True) #随机放大或缩小


train_generator = train_datagen.flow_from_directory(
        imgdir + '/train/',
        target_size=(resize, resize),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1./255,) #测试集不做增强
valid_generator = test_datagen.flow_from_directory(
        imgdir + '/valid/',
        target_size=(resize, resize),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

test_generator = test_datagen.flow_from_directory(
        imgdir + '/test/',
        target_size=(resize, resize),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale')


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# result = model.fit_generator(train_generator,
#           steps_per_epoch=300, 
#           epochs=100, verbose=1,
#           validation_data=valid_generator,
#           validation_steps=25,
#           callbacks=[mcp_save],
#           # max_queue_size=capacity,
#           shuffle = True,
#           workers=1)



# model.save(sys.argv[2])



import datetime
print("{} Start testing...".format(datetime.datetime.now()))

# loss, acc = model.evaluate_generator(test_generator, verbose=2)
# print("pretrained model, test accuracy: {:5.2f}%".format(100*acc))
# print("{} end testing".format(datetime.datetime.now()))

train_generator.reset()
loss, acc = model.evaluate_generator(train_generator, steps=math.ceil(31833/train_batch_size), verbose=2)
# pred = model.predict_generator(train_generator, verbose=1)
# predict_class_indices = np.argmax(pred, axis=1)
# labels = (train_generator.class_indices)
# label = dict((v,k) for k,v in labels.items())

# predictions = [label[i] for i in predict_class_indices]

# print(predictions)
# print(labels)
# print(np.sum(labels == predictions) / len(predictions))
print("pretrained model, training accuracy: {:5.2f}%".format(100*acc))
print("{} end testing".format(datetime.datetime.now()))

valid_generator.reset()
loss, acc = model.evaluate_generator(valid_generator, verbose=2)
print("pretrained model, validation accuracy: {:5.2f}%".format(100*acc))
print("{} end testing".format(datetime.datetime.now()))
