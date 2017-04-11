"""
Created on Thu Mar 30 21:27:38 2017

@author: rudi
"""
import sys
import getopt
import pandas as pd
from sklearn.model_selection import train_test_split
import data_generator as dg

import tensorflow as tf

from keras.layers.core import K
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Activation
from keras.layers import Conv2D, Input
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import model_from_json
import h5py

def nvidia_net():
    # Dropout is used in every FC to prevent the net from overfitting
    keep_prob = 0.7

    model = Sequential()

    # Crop image, normalize it and resize it to the shape that nvidia used too.
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
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

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:sb:e:m:w:j:",["ifile=", "summary", "batch_size", "epochs", "model", "model_weights", "model_to_json"])
    except getopt.GetoptError:
        print('usage: python model.py -i <trainingfiles[,trainingfiles]> [-s]')
        sys.exit(2)

    trainingfiles = []
    batches = 128
    epochs=5

    model_to_json = None
    model_n = None
    # Path to serialized model weights
    mwpath = ''

    for opt, arg in opts:
        if opt == '-h':
            print('usage: python model.py -i <trainingfiles[,trainingfiles]> [-s]')
            sys.exit()
        elif opt in ('-s', '--summary'):
            model = nvidia_net()
            model.summary()
            model = None
            sys.exit(0)
        elif opt in ('-b', '--batch_size'):
            batches = int(arg)
            print('Batch_size: {}'.format(batches))
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
            print('Epochs: {}'.format(epochs))
        elif opt in ('-i', '--ifile'):
            trainingfiles = arg.split(',')
            print( 'Trainingfiles: {}'.format(trainingfiles))
        elif opt in ('-w', '--model_weights'):
            model_weights_path = arg
            print('Path to model_weights: {}'.format(mwpath))
        elif opt in ('-m', '--model'):
            model_n = load_model(arg)
            print('Loaded model: {}'.format(arg))
        elif opt in ('-j', 'model_to_json'):
            with open(arg, 'w') as file:
                model = nvidia_net()
                file.write(model.to_json())
            sys.exit(0)
            

    if len(trainingfiles) == 0:
        print("No training files set")

    tf_frames = [pd.read_csv(tf) for tf in trainingfiles]

    # merge all frames into one
    dataframe = pd.concat(tf_frames)
    # split samples into training and validation samples.
    train_samples, validation_samples = train_test_split(dataframe, test_size=0.2)

    # create training and validation generators.
    # only training samples will be augmented
    train_generator = dg.generator(train_samples, batch_size=batches)
    validation_generator = dg.generator(validation_samples, batch_size=batches, isAugment=False)

    # create the model
    if model_n == None:
        model_n = nvidia_net()

        if mwpath != '':
            model_n.load_weights( mwpath )
        print("Created new model")

    # Tensorboard logging
    callback_tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # Mean square error function is used as loss function because this is a regression problem.
    #model_n.compile(loss='mse', optimizer='adam')
    model_n.compile(optimizer=Adam(lr=0.0001), loss='mse')

    model_n.fit_generator(train_generator,
                          steps_per_epoch = len(train_samples),
                          epochs = epochs,
                          validation_data=validation_generator,
                          validation_steps=len(validation_samples),
                          callbacks=[callback_tb])
    # save model
    model_n.save('./model.h5')

if __name__ == "__main__":
    main(sys.argv[1:])
