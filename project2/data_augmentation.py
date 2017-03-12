# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:54:02 2017

@author: rudi
"""

import cv2
import numpy as np
import random
from numpy import newaxis
import os
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data
base_dir = "traffic-signs-data"

training_file = os.path.join(base_dir, "train.p")
validation_file = os.path.join(base_dir, "valid.p")
testing_file = os.path.join(base_dir, "test.p")

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

def grayscale(image):
    """
    Converts an image to grayscale and reshapes it to (32,32,1)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    
def blur(image, kernel):
    """
    Blurs an image using a provided kernel
    @kernel The kernel used in the gaussian blur function
    """
    blurred_img = cv2.GaussianBlur(image, kernel, 0)
    return blurred_img    
 
def expand(image):
    """
    Adds another axis
    """
    return image[:,:,newaxis]
    
def normalize(image):
    """
    Normalizes the provided image. Image must be greyscale
    """
    image = (image.astype('float') - 128.0) / 128.0
    # Or cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return image

def translate(img, tx, ty):
    """
    Shifts an object into tx and ty direction
    """
    rows,cols = img.shape[:2]  
    M = np.float32([[1,0,tx], [0,1,ty]])
    
    return cv2.warpAffine(img, M, (cols,rows))

def rotate(img, d):
    """
    Rotates an image for d degrees
    @d degrees
    """ 
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), d, 1)
    
    return cv2.warpAffine(img,M,(cols,rows))

def scale(img, s):
    """
    Scales the image
    """
    height, width = img.shape[:2]
    
    return cv2.resize(img,(s*width, s*height), interpolation = cv2.INTER_CUBIC)

def contrast(image, limit):
    """
    Contrast equalization. Can only be applied on gray images
    """
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    cl1 = clahe.apply(image.copy())
    
def contrast_color(image, limit=2.0, gridSize=4):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # equalize the histogram of the Y channel
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # adaptive equalization
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridSize,gridSize))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    # convert the YUV image back to RGB format
    new_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return new_image

def deform_image(image, translation, rotation, contrast):
    """
    Takes an image as input and translates, rotates and scales it based on
    the provided input paramters. Images are generated based on the method presented
    in the paper "A Committee of Neural Networks for Traffic Sign Classification"
    @translation [+|-]T% of the image size for translation
    @rotation    [+|-]R° for rotation
    @scaling     1[+|-]S/100 for scaling
    @contrast    A tuple containing clipping limit and grid size parameters for CLAHE
    """    
    new_image = image.copy()

    # translation
    if translation != 0:
        rows, cols = new_image.shape[:2]
        tx = random.uniform(-translation, translation)
        ty = random.uniform(-translation, translation)
        new_image = translate(new_image, tx, ty)
        
    if rotation != 0:
        d = random.uniform(-rotation, rotation)
        new_image = rotate(new_image, d)
        
    #if scaling != 0:
    #    s = 1 + (random.uniform(-scaling, scaling) / 100.0)
    #    new_image = scale(new_image, s)
    
    if contrast is not None:
        new_image = contrast_color(new_image, contrast[0], contrast[1])        
    return new_image
           
#newimagesA = [deform_image(img, 5, 0, None) for img in X_train] 
#newimagesB = [deform_image(img, 5, 10, (2.0, 4)) for img in X_train] 
#newimagesC = [deform_image(img, 10, 10, (1.0, 2)) for img in X_train]

#import data_plotting as dp

#dp.showimages(newimagesA, y_train, 5, 1, 5, isRandom=False)
#dp.showimages(newimagesB, y_train, 5, 1, 5, isRandom=False)
#dp.showimages(newimagesC, y_train, 5, 1, 5, isRandom=False)

        
        
    