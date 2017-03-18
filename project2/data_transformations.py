# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:54:02 2017

@author: rudi
"""
import os
import math
import cv2
import numpy as np
import random
from numpy import newaxis

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
    Scales the image by a factor s
    @s ... the scaling factor.
    """
    h_old, w_old = img.shape[:2]

    resized_image = cv2.resize(img, None, fx=s, fy=s, interpolation = cv2.INTER_LINEAR)
    h_new, w_new = resized_image.shape[:2]

    if s > 1.0:    
        # crop image with original dimensions from the newly resized one
        (centerY, centerX) = (math.ceil(h_new/2), math.ceil(w_new/2))
        startY = int(centerY - h_old/2)
        startX = int(centerX - w_old/2)
                
        cropped_image = resized_image[startY:startY + h_old, startX:startX + w_old]
        
        height, width = cropped_image.shape[:2]            
        assert(height == 32)
        assert(width == 32)
        
        return cropped_image
        
    elif s < 1.0:
        # New image is smaller than the original one.
        # Pad image with a black border
        bordertop       = int((h_old - h_new) / 2)
        borderbottom    = int((h_old - h_new) / 2) if (h_old - h_new) % 2 == 0 else int((h_old - h_new) / 2) + 1
        borderleft      = int((w_old - w_new) / 2)
        borderright     = int((w_old - w_new) / 2) if (w_old - w_new) % 2 == 0 else int((w_old - w_new) / 2) + 1
        
        padded_image = cv2.copyMakeBorder(resized_image, bordertop, borderbottom, borderleft, borderright, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        height, width = padded_image.shape[:2]        
        assert(height == 32)
        assert(width == 32)
        
        return padded_image

    else:
        return img        
        
def contrast_gray(image, limit):
    """
    Contrast equalization. Can only be applied on gray images
    """
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    return clahe.apply(image.copy())
    
def contrast_color(image, limit=2.0, gridSize=4):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # equalize the histogram of the Y channel
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # adaptive equalization
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridSize,gridSize))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def contrast(image, cfactor=0.5):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = img_yuv[:,:,0] * cfactor
           
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def deform_image(image, translation, rotation, scalefactor, contrast_range):
    """
    Takes an image as input and translates, rotates and scales it based on
    the provided input paramters. Images are generated based on the method presented
    in the paper "A Committee of Neural Networks for Traffic Sign Classification"
    @translation [+|-]T% of the image size for translation
    @rotation    [+|-]R° for rotation
    @scalefactor 1[+|-]S/100 for scaling
    @contrast    A tuple containing clipping limit and grid size parameters for CLAHE
    """    
    new_image = image.copy()

    if translation != 0:
        rows, cols = new_image.shape[:2]
        tx = random.uniform(-translation, translation)
        ty = random.uniform(-translation, translation)
        new_image = translate(new_image, tx, ty)
        
    if rotation != 0:
        d = random.uniform(-rotation, rotation)
        new_image = rotate(new_image, d)

    if contrast_range is not None:
        assert(len(contrast_range) == 2)
        assert(contrast_range[0] < contrast_range[1])
        
        cfactor = random.uniform(contrast_range[0], contrast_range[1])
        new_image = contrast(new_image, cfactor)  
        
    if scalefactor != 0:
        s = 1 + (random.uniform(-scalefactor, scalefactor) / 100.0)
        new_image = scale(new_image, s)
          
    return new_image

#import data_plotting as dp

#dp.set_plotsize(6,4)
#dp.showimages([X_train[0]], [y_train[0]], 1, 1, 1, isRandom=False)

#img = contrast(X_train[0])
#dp.showimages([img], [y_train[0]], 1, 1, 1, isRandom=False)

#img = scale(X_train[0], 0.7)
#for i in range(10):
#   img = deform_image(X_train[0], 5, 10, 30, [0.5, 1.5])
#   dp.showimages([img], [y_train[0]], 1, 1, 1, isRandom=False)


#newimagesA = [deform_image(img, 5, 0, None) for img in X_train] 
#newimagesB = [deform_image(img, 5, 10, (2.0, 4)) for img in X_train] 
#newimagesC = [deform_image(img, 10, 10, (1.0, 2)) for img in X_train]

#import data_plotting as dp

#dp.showimages(newimagesA, y_train, 5, 1, 5, isRandom=False)
#dp.showimages(newimagesB, y_train, 5, 1, 5, isRandom=False)
#dp.showimages(newimagesC, y_train, 5, 1, 5, isRandom=False)

        
        
    