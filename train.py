# trains the joker network
# dataset specified by TRAIN_DATA_FOLDER in globals_and_utils
# this folder contains train/ valid/ test/ folders each with class1 (nonjoker) and class2 (joker) examples
# see dataset_utils for methods to create the training split folders
# author: Tobi Delbruck
import argparse
import glob

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

from globals_and_utils import *
import datetime

INITIALIZE_MODEL_FROM_LATEST=True # set True to initialize weights to latest saved model

log=my_logger(__name__)
LOG_FILE='training.log'
fh=logging.FileHandler(LOG_FILE,'w') # 'w' to overwrite, not append
fh.setLevel(logging.INFO)
fmtter=logging.Formatter(fmt= "%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(fmtter)
log.addHandler(fh)

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

args=None # if run as module
model_folder=None
tflite_model_name=None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train:train CNN for txsy', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--test_accuracy", action='store_true',
        help="run network test rather than train")
    args = parser.parse_args()


start_time=time.time()
start_timestr = time.strftime("%Y%m%d-%H%M")
log.info(f'Tensorflow version {tf.version.VERSION}')
log.info(f'dataset path: TRAIN_DATA_FOLDER={TRAIN_DATA_FOLDER}')

num_classes=2
checkpoint_filename='joker_net_checkpoint.hdf5'

model = create_model()
if INITIALIZE_MODEL_FROM_LATEST:
    existing_model_folders=glob.glob(JOKER_NET_BASE_NAME+ '*/')
    TIMEOUT=30
    if len(existing_model_folders)>0:
        latest_model_folder=max(existing_model_folders, key=os.path.getmtime)
        getmtime_checkpoint = os.path.getmtime(checkpoint_filename)
        getmtime_stored_model= os.path.getmtime(latest_model_folder)
        if os.path.isfile(checkpoint_filename)\
                and getmtime_checkpoint > getmtime_stored_model \
                and yes_or_no(f'checkpoint {checkpoint_filename} modified { datetime.datetime.fromtimestamp(getmtime_checkpoint)} \nis newer than saved model {existing_model_folders[0]} modified {datetime.datetime.fromtimestamp(getmtime_stored_model)},\n start from it?',timeout=TIMEOUT):
            log.info(f'loading weights from checkpoint {checkpoint_filename}')
            model.load_weights(checkpoint_filename)
        else:
            if yes_or_no(f'model {latest_model_folder} exists, start from it?',timeout=TIMEOUT):
                log.info(f'initializing model from {latest_model_folder}')
                model=load_model(latest_model_folder)
            else:
                log.info('creating new empty model')
                model = create_model()
    else:
        if os.path.isfile(checkpoint_filename) and yes_or_no('checkpoint exists, start from it?',timeout=TIMEOUT):
            log.info(f'loading weights from checkpoint {checkpoint_filename}')
            model = create_model()
            model.load_weights(checkpoint_filename)
        else:
            yn=yes_or_no("Could not find saved model or checkpoint. Initialize a new model?",timeout=TIMEOUT)
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
train_datagen = ImageDataGenerator( #实例化
    rescale=1./255, # todo check this
    rotation_range = 30,  #图片随机转动的角度
    width_shift_range = 0.3, #图片水平偏移的幅度
    height_shift_range = 0.3, #图片竖直偏移的幅度
    zoom_range = 0.2,
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

def print_datagen_summary(gen: ImageDataGenerator):
    nsamp=gen.samples
    num_joker=np.count_nonzero(gen.labels)
    num_nonjoker=nsamp-num_joker
    def pc(n):
        return 100*float(n)/nsamp
    log.info(f'summary of {gen.directory}:'
         f' {gen.samples} samples:\t{pc(num_nonjoker):.1f}% nonjoker,\t{pc(num_joker):.1f}% joker\n')

print_datagen_summary(train_generator)
print_datagen_summary(valid_generator)
print_datagen_summary(test_generator)

stop_early = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
save_checkpoint = ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')
# Profile from batches 10 to 15
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tb_profiler_log',
                                             profile_batch='10, 15')

if args is None or not args.test_accuracy:
    log.info('starting training')
    history=None
    try:
        history = model.fit(train_generator, # todo back to train generator
                  # steps_per_epoch=150,
                  epochs=300, verbose=1,
                  validation_data=valid_generator,
                  # validation_steps=25,
                  callbacks=[stop_early, save_checkpoint,tb_callback],
                  # max_queue_size=capacity,
                  shuffle = True,
                  workers=1)
        if history is not None:
            try:
                training_history_filename='training_history.npy'
                np.save(training_history_filename,history.history)
                log.info(f'Done with model.fit; history is \n{history.history} and is saved as {training_history_filename}')
                log.info(f'history.history.keys()={history.history.keys()}')

                # summarize history for accuracy
                plt.plot(history.history['categorical_accuracy'])
                plt.plot(history.history['val_categorical_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
            except KeyError as k:
                log.warning('could not plot, caught {k}, history.history.keys()={history.history.keys()} ')
    except KeyboardInterrupt:
        log.warning('keyboard interrupt, saving model and testing')

    model_folder= f'{JOKER_NET_BASE_NAME}_{start_timestr}'

    log.info(f'saving model to folder {model_folder}')
    model.save(model_folder)

    log.info('converting model to tensorflow lite model')
    converter = tf.lite.TFLiteConverter.from_saved_model(model_folder) # path to the SavedModel directory
    tflite_model = converter.convert()
    tflite_model_name=f'{model_folder}.tflite'

    log.info(f'saving tflite model as {tflite_model_name}')
    with open(tflite_model_name, 'wb') as f:
      f.write(tflite_model)
    # end training part

log.info('evaluating accuracy')
# test_generator.reset()
gen=test_generator
gen.reset()
loss, acc = model.evaluate(gen, verbose=1)
log.info(f'On test {gen}  loss={loss:.3f}, acc={acc:.4f}')
gen.reset()
y_output = model.predict(gen, verbose=1) # matrix of Nx2 with each row being the nonjoker/joker score
y_pred=np.argmax(y_output,axis=1) # vector of predicted classes 0/1 for nonjoker/joker
y_true=gen.labels # vector of ground truth classes, 0 or 1 for nonjoker/joker
balanced_accuracy=balanced_accuracy_score(y_true,y_pred)
conf_matrix=confusion_matrix(y_true, y_pred)
log.info(f'**** final test set balanced accuracy: {balanced_accuracy*100:6.3f}\% (chance would be 50\%)\nConfusion matrix nonjoker/joker:\n {conf_matrix}')
np.set_printoptions(precision=2)
# disp = plot_confusion_matrix(classifier, X_test, y_test,
#                                  display_labels=['nonjoker','joker'],
#                                  cmap=plt.cm.Blues,
#                                  normalize=True)
# disp.ax_.set_title('joker/nonjoker confusion matrix')

elapsed_time_min=(time.time()-start_time)/60
log.info(f'**** done training after {elapsed_time_min:4.1f}m; model saved in {model_folder} and {tflite_model_name}.'
         f'\nSee {LOG_FILE} for logging output for this run.')

