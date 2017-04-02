# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:27:38 2017

@author: rudi
"""
from keras.layers.core import K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.pooling import AveragePooling2D
from keras.layers import Cropping2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization

def nvidia_net():
    keep_prob = 0.5
    keep_prob_fc = 0.5

    model = Sequential()

    # pre-processing
    model.add(Cropping2D(cropping=((40,25), (0,0)), input_shape=(160,320,3)))
    
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200)))) 
    
    model.add(Lambda(lambda x: (x - 128.0) / 128.0)) # normalization
        
    # Convnet
    
    model.add(Conv2D(24, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(36, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(48, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(keep_prob))

    # FC
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(50))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(10))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(1))

    return model

import pandas as pd
import data_generator as dg
df = pd.read_csv('data/driving_log.csv')

# Split samples into training and validation samples.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(df, test_size=0.2)

train_generator = dg.generator(train_samples, batch_size=32)
validation_generator = dg.generator(validation_samples, batch_size=32)


model_n = nvidia_net()
model_n.summary()

# Mean square error function is used as loss function because this 
# is a regression problem.
model_n.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, 
#     validation_data=None, validation_steps=None, class_weight=None, 
#     max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)

model_n.fit_generator(train_generator, 
                      steps_per_epoch = len(train_samples),
                      epochs = 1,
                      validation_data=validation_generator,
                      validation_steps=len(validation_samples))

# save model
model_n.save('./model.h5')