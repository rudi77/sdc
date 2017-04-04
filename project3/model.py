# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:27:38 2017

@author: rudi
"""
import sys
import getopt
import pandas as pd
from sklearn.model_selection import train_test_split
import data_generator as dg

from keras.layers.core import K
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Activation
from keras.layers import Conv2D, Input
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

def nvidia_net():
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
    
    # Dropout is used in every FC to prevent the net from overfitting
    keep_prob = 0.7
    
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
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('usage: python model.py -i <trainingfiles[,trainingfiles]>')
        sys.exit(2)

    trainingfiles = []      
    for opt, arg in opts:
        if opt == '-h':
            print('usage: python model.py -i <trainingfiles[,trainingfiles]>')
            sys.exit()
        elif opt in ('-i', '--ifile'):
            trainingfiles = arg.split(',')
            print( 'Trainingfiles: {}'.format(trainingfiles))
   
    if len(trainingfiles) == 0:
        print("No training files set")
   
    tf_frames = [pd.read_csv(tf) for tf in trainingfiles]
   
    # merge all frames into one
    dataframe = pd.concat(tf_frames)
    # split samples into training and validation samples.
    train_samples, validation_samples = train_test_split(dataframe, test_size=0.2)
    
    # create training and validation generators.
    # only training samples will be augmented
    train_generator = dg.generator(train_samples, batch_size=128)
    validation_generator = dg.generator(validation_samples, batch_size=128, isAugment=False)
    
    # create the model
    model_n = nvidia_net()
    model_n.summary()
    
    # Tensorboard logging
    callback_tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    
    # Mean square error function is used as loss function because this is a regression problem.
    model_n.compile(loss='mse', optimizer='adam')
    
    model_n.fit_generator(train_generator, 
                          steps_per_epoch = len(train_samples) * 3,
                          epochs = 10,            
                          validation_data=validation_generator,
                          validation_steps=len(validation_samples),
                          callbacks=[callback_tb])
    # save model
    model_n.save('./model.h5')  
   
if __name__ == "__main__":
    main(sys.argv[1:])
