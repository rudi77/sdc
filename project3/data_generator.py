# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:22:00 2017

@author: rudi
"""
import os
import re
import cv2
import numpy as np
import sklearn

#angle_offset = 0.06
angle_offset = 0.25

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
    
    # Assumption: On the ec2 instance all images are placed into the ./data/IMG folder.
    # The path is split into tokens and only the file name of the image as used as the path
    # to the image may differ on your host machine. This should work for windows as well as
    # linux paths.
    tokens = re.split('[\\,\\\\,/]', sample.iat[0,camera])
    path = os.path.join('data', 'IMG')
    path = os.path.join(path, tokens[-1])
    
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

def generator(samples, batch_size, isAugment = True):
    """
    Generates batches of training sets.
    @samples       The samples, as a pandas dataframe, which shall be split into batches
    @batch_size    The size of one batch.
    @isAugment     If set to true then the samples will be augmented, i.e. left and right camera images 
                    will be taken into account too.
    """
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        
        for batch_samples in chunker(samples,batch_size):
           
            images = []
            angles = []
            
            for batch_sample in chunker(batch_samples, 1):
                # Center image
                center_image, center_angle = image_and_angle(batch_sample, 0)
                images.append(center_image)
                angles.append(center_angle)
            
                if isAugment:
                    # Left image
                    left_image, left_angle = image_and_angle(batch_sample, 1)
                    images.append(left_image)
                    angles.append(left_angle + angle_offset)
                    
                    # Right image
                    right_image, right_angle = image_and_angle(batch_sample, 2)
                    images.append(right_image)
                    angles.append(right_angle - angle_offset)
                    
                    # Flip image horizontally, also invert sign of steering angle
                    if center_angle != 0.0:
                        center_flipped_image, center_flipped_angle = flip_image(center_image, center_angle)
                        images.append(center_flipped_image)
                        angles.append(center_flipped_angle)
                        
                        #left_flipped_image, left_flipped_angle = flip_image(left_image, left_angle)
                        #images.append(left_flipped_image)
                        #angles.append(left_flipped_angle)
                        
                        #right_flipped_image, right_flipped_angle = flip_image(right_image, right_angle)
                        #images.append(right_flipped_image)
                        #angles.append(right_flipped_angle)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train
