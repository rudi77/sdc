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

def nvidia_net():
    # Dropout is used in every FC to prevent the net from overfitting
    keep_prob = 0.7

    model = Sequential()

    # Crop image, normalize it and resize it to the shape that nvidia used too.
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,1)))
    model.add(Lambda(lambda x: x / 255 - 0.5))
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

def rudi_net():
    # Dropout is used in every FC to prevent the net from overfitting
    keep_prob = 0.7

    model = Sequential()

    # Crop image, normalize it and resize it to the shape that nvidia used too.
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda x: (x - 128.) / 128.))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200))))

    # Convolutional Layers
    model.add(Conv2D(24, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

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

def usage():
    print("usage: python model.py -i training_log.csv")
    print("optional arguments")
    print("-h                   Help")
    print("-s, --summary=       Model layer and parameter summary for a certain model, either 'nvidia' or 'rudi'")
    print("-a, --arch=          The model that shall be used. Either 'nvidia' or 'rudi'. Default is 'nvidia'")    
    print("-b, --batch_size=    Batch size")
    print("-e, --epochs=        Number of epochs")
    print("-m, --model=         Load a stored model")
    print("-j, --model_to_json= Write architecture to a json file")
    print("-p, --initial_epoch= Set the initial epoch. Useful when restoring a saved model")
    

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
        self.architecture = "nvidia"
        
    def show_summary(self):
        print('Training files: {}'.format(self.trainingfiles))
        print('Batch_size: {}'.format(self.batches))
        print('Epochs: {}'.format(self.epochs))
        print('Initial epoch: {}'.format(self.initial_epoch))
        print('model_to_json: {}'.format(self.model_to_json))
        print('architecture: {}'.format(self.architecture))
    
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
            model = None
            if arg == 'nvidia':
                model = nvidia_net()
            else:
                model = rudi_net()
            model.summary()
            sys.exit(0)
        elif opt in ('-i', '--ifile'):
            settings.trainingfiles = arg.split(',')            
        elif opt in ('-a', '--arch'):
            if arg != "nvidia" and arg != "rudi":
                print("network architecture {} not supported".format(settings.architecture))
                sys.exit(0)
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
                model = nvidia_net()
                file.write(model.to_json())
            sys.exit(0)
            
    settings.show_summary()
    
    return settings
    
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:s:b:e:m:j:p:a:",["ifile=", "summary=", "batch_size=", "epochs=", "model=", "model_to_json=", "initial_epoch=", "arch="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    settings = parse_options(opts)
            

    if len(settings.trainingfiles) == 0:
        print("No training files set")
        sys.exit(0)


    print("reading training files")
    tf_frames = [pd.read_csv(trainfile) for trainfile in settings.trainingfiles]

    # merge all frames into one
    dataframe = pd.concat(tf_frames)
    # split samples into training and validation samples.
    train_samples, validation_samples = train_test_split(dataframe, test_size=0.2)

    # create training and validation generators.
    # only training samples will be augmented
    train_generator = dg.generator(train_samples, batch_size=settings.batches)
    validation_generator = dg.generator(validation_samples, batch_size=settings.batches, isAugment=False)

    # create the model
    if settings.model == None:
        if settings.architecture == "nvidia":
            settings.model = nvidia_net()
        elif settings.architecture == "rudi":
            settings.model = rudi_net()

        print("Created new model")

    # Tensorboard logging
    callback_tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    
    # checkpoint
    filepath="checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [callback_tb, checkpoint]

    # Mean square error function is used as loss function because this is a regression problem.
    settings.model.compile(optimizer="adam", loss='mse')

    print("start training")
    settings.model.fit_generator(train_generator,
                          steps_per_epoch = len(train_samples) / settings.batches,
                          epochs = settings.epochs,
                          validation_data=validation_generator,
                          validation_steps=len(validation_samples) / settings.batches,
                          callbacks=callbacks_list,
                          initial_epoch=settings.initial_epoch)

if __name__ == "__main__":
    main(sys.argv[1:])
