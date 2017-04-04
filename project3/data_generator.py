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

#steering_angle_left_right = 0.061
steering_angle_left_right = 0.25
# Taken from Stackoverflow:
# http://stackoverflow.com/questions/25699439/how-to-iterate-over-consecutive-chunks-of-pandas-dataframe-efficiently
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def image_and_angle(sample, camera):
    tokens = re.split('[\\\\,/]', sample.iat[0,camera])
    path = os.path.join('data', 'IMG')
    path = os.path.join(path, tokens[-1])
    
    image = cv2.imread(path)
    angle = float(sample.iat[0,3])
    
    return (image, angle)


def generator(samples, batch_size, isAugment = True):
    num_samples = len(samples)
        
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
                    angles.append(left_angle + steering_angle_left_right)
                    
                    # Right image
                    right_image, right_angle = image_and_angle(batch_sample, 2)
                    images.append(right_image)
                    angles.append(right_angle - steering_angle_left_right)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
