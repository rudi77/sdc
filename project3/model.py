"""
Created on Thu Mar 30 21:27:38 2017

@author: rudi
"""
import sys
import getopt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import data_generator as dg

import tensorflow as tf

from keras.layers.core import K
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Activation
from keras.layers import Conv2D, Input
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
from keras import optimizers
import h5py

def create_model():
    # Dropout is used in every FC to prevent the net from overfitting
    keep_prob = 0.3

    model = Sequential()

    # Crop image, normalize it and resize it to the shape that nvidia used too.
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x, name=None)))
    model.add(Lambda(lambda x: (x - 128.) / 128.))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200))))

    # Convolutional Layers
    model.add(Conv2D(24, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(48, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))

    return model

def split_samples(trainingfiles, test_size=0.2):
    """
    Reads all training_log.csv files into dataframe and finally creates
    a training and validation set.
    @trainingfiles  a list of all training_log.csv files which shall be used for data sets generation.
    @test_size      
    """
    header = ["center","left", "right", "steering", "throttle", "brake", "speed"]
    
    dataframe = None
    
    for trainfile in trainingfiles:
        # driving_log.csv in data_base contains a header. The others do not. Therefore, add
        # a header otherwise dataframe.append(df) does not work - at least I do not know how I could append
        # a dataframe to an existing one without a header.
        df = pd.read_csv(trainfile) if "data_base" in trainfile else pd.read_csv(trainfile, names=header)

        if dataframe is None:
            dataframe = df
        else:
            dataframe = dataframe.append(df)
    
    print("Total number of sample rows {}".format(len(dataframe)))
    
    # split samples into training and validation samples.
    train_samples, validation_samples = train_test_split(dataframe, test_size=test_size)
    
    return train_samples, validation_samples
    

def usage():
    print("usage: python model.py -i training_log.csv")
    print("optional arguments")
    print("-h                   Help")
    print("-s, --summary        Model layer and parameter summary")   
    print("-b, --batch_size=    Batch size")
    print("-p, --initial_epoch= Set the initial epoch. Useful when restoring a saved model.")    
    print("-e, --epochs=        Number of epochs")
    print("-m, --model=         Load a stored model")
    print("-j, --model_to_json= Write architecture to a json file")
    

class Settings:
    """
    Keeps all necessary settings needed to train a model
    """
    def __init__(self):
        self.trainingfiles = []
        self.batches        = 128
        self.epochs=5
        self.initial_epoch = 0
        self.model_to_json = None
        self.model = None
        
    def show_summary(self):
        print('Training files: {}'.format(self.trainingfiles))
        print('Batch_size: {}'.format(self.batches))
        print('Epochs: {}'.format(self.epochs))
        print('Initial epoch: {}'.format(self.initial_epoch))
        print('model_to_json: {}'.format(self.model_to_json))
    
def parse_options(opts):
    """
    Parses command line options and retuns a settings object which contains
    all settings needed to train a model.
    """
    settings = Settings()
    
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ('-s', '--summary'):
            model = create_model()
            model.summary()
            sys.exit(0)
        elif opt in ('-i', '--ifile'):
            settings.trainingfiles = arg.split(',')            
        elif opt in ('-a', '--arch'):
            settings.architecture = arg            
        elif opt in ('-b', '--batch_size'):
            settings.batches = int(arg)
        elif opt in ('-e', '--epochs'):
            settings.epochs = int(arg)
        elif opt in ('-p', '--initial_epoch'):
            settings.initial_epoch = int(arg)
        elif opt in ('-m', '--model'):
            settings.model = load_model(arg)
        elif opt in ('-j', 'model_to_json'):
            with open(arg, 'w') as file:
                model = create_model()
                file.write(model.to_json())
            sys.exit(0)
            
    settings.show_summary()
        
    return settings

def check_settings(settings):
    if len(settings.trainingfiles) == 0:
        print("No training files set")
        sys.exit(0)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:sb:e:m:j:p:a:",["ifile=", "summary", "batch_size=", "epochs=", "model=", "model_to_json=", "initial_epoch=", "arch="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    settings = parse_options(opts)
    check_settings(settings)
            
    train_samples, validation_samples = split_samples(settings.trainingfiles)
    
    # create training and validation generators - only training samples will be augmented
    train_generator = dg.generator(train_samples, batch_size=settings.batches)
    validation_generator = dg.generator(validation_samples, batch_size=settings.batches, isAugment=False)

    # Tensorboard logging
    callback_tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    
    # checkpoint
    filepath="new_model-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [callback_tb, checkpoint]

    # create model
    settings.model = create_model() if settings.model == None else settings.model

    # Mean square error function is used as loss function because this is a regression problem.
    settings.model.compile(optimizer="adam", loss='mse')

    print("start training")
    history = settings.model.fit_generator(train_generator,
                          steps_per_epoch = len(train_samples) / settings.batches,
                          epochs = settings.epochs,
                          validation_data=validation_generator,
                          validation_steps = len(validation_samples) / settings.batches,
                          callbacks=callbacks_list,
                          initial_epoch=settings.initial_epoch,
                          verbose=1)
    
    dg.plot_history(history)

if __name__ == "__main__":
    main(sys.argv[1:])
