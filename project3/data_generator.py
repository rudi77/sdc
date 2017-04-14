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
import sklearn

angle_offset = 0.25

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
                        
                    # Change contrast
                    img_brightness = brightness(center_image, random.uniform(0.4, 1.2))
                    images.append(img_brightness)
                    angles.append(center_angle)
                            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train

# Taken from Stackoverflow:
# http://stackoverflow.com/questions/25699439/how-to-iterate-over-consecutive-chunks-of-pandas-dataframe-efficiently
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def crop_image(image, x, wx, y, hy):
    return image[y:hy, x:wx]

def to_yuv(image):
    """
    Converts and RGB to an YUV image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV    )

def brightness(image, cfactor=0.5):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] =  np.where(img_yuv[:,:,0] * cfactor < 255, img_yuv[:,:,0] * cfactor, 255) 
            
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def add_shadow(image):
    w,h,c = image.shape          
    return image

def grayscale(image):
    """
    Converts an image to grayscale and reshapes it to (32,32,1)
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return img_gray[:,:,newaxis]

def rotate(img, d):
    """
    Rotates an image for d degrees
    @d degrees
    """ 
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), d, 1)
    
    return cv2.warpAffine(img,M,(cols,rows))

def shift_horizontal(img, angle):
    """
    Shifts an image in horizontal direction.
    @image  The image to be shifted.
    @angle  Original steering angle which is updated.
    """
    
    shiftx = random.randint(-50, 50)
    #angle_new = angle_offset * shiftx + angle_old
    rows,cols = img.shape[:2]  
    M = np.float32([[1, 0, shiftx], [0, 1, 0]])
    
    return cv2.warpAffine(img, M, (cols,rows)), angle

def flip_image(image, steering_angle):
    """
    Flips an image horizontally and inverts the given steering angle.
    """
        
    image_flipped = np.fliplr(image)
    flipped_angle = -steering_angle
    
    return image_flipped, flipped_angle


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