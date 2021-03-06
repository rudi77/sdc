# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:22:00 2017

@author: rudi
"""
import os
import re
import random
import cv2
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

def generator(samples, batch_size, isAugment = True):
    """
    Generates batches of data sets.
    @samples       The samples, as a pandas dataframe, which shall be split into batches
    @batch_size    The size of one batch.
    @isAugment     If set to true then the samples will be augmented, i.e. left and right camera images 
                    will be taken into account too.
    """
    CENTER = 0
    LEFT = 1
    RIGHT = 2
    
    angle_offset = 0.25
    
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        
        for batch_samples in chunker(samples,batch_size):
           
            images = []
            angles = []
            
            for batch_sample in chunker(batch_samples, 1):
                
                # Center image
                center_image, center_angle = image_and_angle(batch_sample, CENTER)                              
                images.append(center_image)
                angles.append(center_angle)
            
                if isAugment:
                    # Left image
                    left_image, left_angle = image_and_angle(batch_sample, LEFT)
                    images.append(left_image)
                    angles.append(left_angle + angle_offset)
                    
                    # Right image
                    right_image, right_angle = image_and_angle(batch_sample, RIGHT)
                    images.append(right_image)
                    angles.append(right_angle - angle_offset)
                    
                    # Flip image horizontally, also invert sign of steering angle
                    if center_angle != 0.0:
                        center_flipped_image, center_flipped_angle = flip_image(center_image, center_angle)
                        images.append(center_flipped_image)
                        angles.append(center_flipped_angle)
                        
                    # Change brightness
                    img_brightness = brightness(center_image, random.uniform(0.2, 1.2))
                    images.append(img_brightness)
                    angles.append(center_angle)
                            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train

# Taken from Stackoverflow:
# http://stackoverflow.com/questions/25699439/how-to-iterate-over-consecutive-chunks-of-pandas-dataframe-efficiently
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def image_and_angle(sample, camera):
    """
    Reads image with its corresponding steering angle and returns
    it as a tuple (image, steering_angle)
    @sample represents one row in the csv file from the image path and steering angle 
            is extracted.
    @camera an int [0,1,2] representing the camera image that shall be loaded.
            Where 0...center, 1...left and 2...right camera image.                   
    """
    
    # Assumption: On the ec2 instance all images are placed into the ./data_xxx/IMG folder.
    # The path is split into tokens and only the file name of the image is used as the path
    # to the image may differ on your host machine. This should work for windows as well as
    # linux paths.

    # Get the relative path to the image.    
    relpath = sample.iat[0,camera].split('data')[-1].strip()
    relpath = 'data_base\\' + relpath if relpath.startswith("IMG") else 'data' + relpath
    
    # Split relative path and rejoin them again. With this approach
    # we should be able to handle windows as well as linux path strings.
    tokens = re.split('[\\,\\\\,/]', relpath)
    path = os.path.join(*tokens)
    
    image = cv2.imread(path)
    angle = float(sample.iat[0,3])
    
    return (image, angle)

def flip_image(image, steering_angle):
    """
    Flips an image horizontally and inverts the given steering angle.
    """
        
    image_flipped = np.fliplr(image)
    flipped_angle = -steering_angle
    
    return image_flipped, flipped_angle

def brightness(image, cfactor=0.5):
    """
    Changes the brightness of every pixel of the image by a certain factor
    @image      the images whose brightness is changed
    @cfactor    brightness factor
    """
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] =  np.where(img_yuv[:,:,0] * cfactor < 255, img_yuv[:,:,0] * cfactor, 255) 
            
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def shift_horizontal(img, angle):
    """
    Shifts an image in horizontal direction.
    @image  The image to be shifted.
    @angle  Original steering angle which is updated.
    """
    # Linear equation is used to calculate the offset that shall
    # be added to the original angle.
    # offset = k * shiftx + d
    k = 0.005
    
    shiftx = random.randint(-50, 50)
    angle_new = k * shiftx + angle

    # check value of new angle. must be in the interval -1 and 1
    if angle_new < -1.0:
        angle_new = -1.0
    if angle_new > 1.0:
        angle_new = 1.0
    
    rows,cols = img.shape[:2]  
    M = np.float32([[1, 0, shiftx], [0, 1, 0]])
    
    return cv2.warpAffine(img, M, (cols,rows)), angle

def plot_history( history, filename='./loss.png' ):
    """
    Plot the training and validation loss for each epoch
    @history contains the training and validation loss values for a full run.
    """
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(filename)