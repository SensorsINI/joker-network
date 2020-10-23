from __future__ import absolute_import, division, print_function, unicode_literals

import math
import tensorflow as tf
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from globals_and_utils import *

log=my_logger(__name__)
log.info(f'Tensorflow version {tf.version.VERSION}')

resize = IMSIZE
imgdir = TRAIN_DATA_FOLDER
log.info(f'dataset path: TRAIN_DATA_FOLDER={imgdir}')


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


num_classes=2
def load_data():
    import os, os.path
    dataset_size=len(os.listdir(imgdir))
    testsize=math.floor(dataset_size*.1)

    log.info('---------------', dataset_size)
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
    X = np.zeros((len(paths),img_size,img_size,color_type),dtype=np.uint8)
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
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model


model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

train_batch_size = 128
valid_batch_size = 64
test_batch_size = 32

log.info('making training generator')
train_datagen = ImageDataGenerator( #实例化
    rescale=1./255, # todo check this
    rotation_range = 30,  #图片随机转动的角度
    width_shift_range = 0.3, #图片水平偏移的幅度
    height_shift_range = 0.3, #图片竖直偏移的幅度
    zoom_range = 0.5,
    horizontal_flip=True) #随机放大或缩小

train_generator = train_datagen.flow_from_directory(
        imgdir + '/train/',
        target_size=(resize, resize),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

log.info('making validation generator')
test_datagen = ImageDataGenerator(rescale=1./255,) #测试集不做增强
valid_generator = test_datagen.flow_from_directory(
        imgdir + '/valid/',
        target_size=(resize, resize),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

log.info('making test generator')
test_generator = test_datagen.flow_from_directory(
        imgdir + '/test/',
        target_size=(resize, resize),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale')


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint('model_checkpoint.hdf5', save_best_only=True, monitor='val_loss', mode='min')

log.info('starting training')
result = model.fit(train_generator,
          steps_per_epoch=300, 
          epochs=100, verbose=1,
          validation_data=valid_generator,
          validation_steps=25,
          # callbacks=[mcp_save],
          # max_queue_size=capacity,
          shuffle = True,
          workers=1)



timestr = time.strftime("%Y%m%d-%H%M")
model_folder= f'joker_net_{timestr}'

log.info(f'saving model to folder {model_folder}')
model.save(model_folder)

log.info('converting model to tensorflow lite model')
converter = tf.lite.TFLiteConverter.from_saved_model(model_folder) # path to the SavedModel directory
tflite_model = converter.convert()
tflite_model_name=f'{model_folder}.tflite'

log.info(f'saving tflite model as {tflite_model_name}')
with open(tflite_model_name, 'wb') as f:
  f.write(tflite_model)

log.info('evaluating test set accuracy')
test_generator.reset()
loss, acc = model.evaluate(test_generator, verbose=2)
log.info("test set accuracy: {:5.2f}%".format(100*acc))

log.info('done training')


