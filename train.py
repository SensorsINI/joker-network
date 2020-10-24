# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train valid test folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
import glob

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

from globals_and_utils import *

INITIALIZE_MODEL_FROM_LATEST=True # set True to initialize weights to latest saved model

log=my_logger(__name__)
log.info(f'Tensorflow version {tf.version.VERSION}')
log.info(f'dataset path: TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}')

num_classes=2

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
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model

if INITIALIZE_MODEL_FROM_LATEST:
    existing_model_folders=glob.glob(JOKER_NET_BASE_NAME+ '*/')
    if len(existing_model_folders)>0:
        latest_model_folder=max(existing_model_folders, key=os.path.getmtime)
        log.info(f'initializing model from {latest_model_folder}')
        model=load_model(latest_model_folder)
    else:
        log.error('could not find saved model; set INITIALIZE_MODEL_FROM_LATEST to False')
        quit(1)
else:
    log.info('creating new empty model')
    model = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
model.summary()

train_batch_size = 64
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

log.info('making training generator')
train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/train/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

log.info('making validation generator')
valid_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/valid/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

log.info('making test generator')
test_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_FOLDER + '/test/',
        target_size=(IMSIZE, IMSIZE),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale')


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
stop_early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
checkpoint_filename='joker_net_checkpoint.hdf5'
save_checkpoint = ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')

if os.path.isfile(checkpoint_filename):
    log.info(f'loading weights from checkpoint {checkpoint_filename}')
    model.load_weights(checkpoint_filename)

log.info('starting training')
history = model.fit(train_generator,
          # steps_per_epoch=150,
          epochs=300, verbose=1,
          validation_data=valid_generator,
          # validation_steps=25,
          callbacks=[stop_early, save_checkpoint],
          # max_queue_size=capacity,
          shuffle = True,
          workers=1)

log.info(f'Done with model.fit; history is \n{history.history}')

timestr = time.strftime("%Y%m%d-%H%M")
model_folder= f'{JOKER_NET_BASE_NAME}_{timestr}'

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

log.info(f'done training; model saved in {model_folder} and {tflite_model_name}')


