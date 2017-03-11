# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:54:02 2017

@author: rudi
"""

import matplotlib.pyplot as plt
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
    #img_gray_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    #img_gray_blur = img_gray_blur[:, :, newaxis]
    #return img_gray_blur
 
def expand(image):
    return image[:,:,newaxis]
    
def normalize(image):
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
    
    """
    height, width = img.shape[:2]
    
    return cv2.resize(img,(s*width, s*height), interpolation = cv2.INTER_CUBIC)

def deform_image(image, translation, rotation):
    """
    Takes an image as input and translates, rotates and scales it based on
    the provided input paramters. Images are generated based on the method presented
    in the paper "A Committee of Neural Networks for Traffic Sign Classification"
    @translation [+|-]T% of the image size for translation
    @rotation    [+|-]R° for rotation
    @scaling     1[+|-]S/100 for scaling
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
        
    return new_image
       

plt.rcParams['figure.figsize'] = (12, 10)
def showimages(images):
    for i in range(9):
    		rand_idx = np.random.randint(len(images))
    		image = images[rand_idx]
    		plt.subplot(3, 3, i+1)
    		plt.imshow(image)
    		plt.title('Image Idx: %d' % (rand_idx,))
    
newimagesA = [deform_image(img, 5, 0) for img in X_train] 
newimagesB = [deform_image(img, 5, 10) for img in X_train] 
newimagesC = [deform_image(img, 10, 10) for img in X_train]


# Convert each image to grayscale
#X_train_gray = [to_grayscale(image.copy()) for image in X_train]
#X_valid_gray = [to_grayscale(image.copy()) for image in X_valid]

#plot_traffic_signs_of_class(classes_dict[0], X_train_gray)
# Normalize images
#X_train_normalized = [normalize(image) for image in X_train_gray]
#X_valid_normalized = [normalize(image) for image in X_valid_gray]

#X_train = X_train_normalized
#X_valid = X_valid_normalized   


showimages(newimagesA)
#showimages(newimagesB)
#showimages(newimagesC)

        
        
    