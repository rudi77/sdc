# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:18:03 2017

@author: rudi
"""
import os
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
    
# Our data set consists of vehicle and non-vehicle images. We iterate over all images, apply HOG to each image and store
# HOG features with a corresponding label, 1 for car and 0 for non-car, as feature row in a pandas data frame. Finally,
# we write the data frame in a file in csv format.

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    rhist = np.histogram(img_hsv[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img_hsv[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img_hsv[:,:,2], bins=nbins, range=bins_range)
    
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1]) / 2
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
    

def create_header(num_features, pix_per_cell, cells_per_block, orient):
    """
    Creates the header for our dataset.
    """
    header_entries = ['hog_' + str(pix_per_cell) + '_' + str(cells_per_block) + '_' + str(orient) + '_' + str(i) for i in range(num_features)]    
    header_entries.insert(0,'imagename')
    header_entries.append('label')
    
    return header_entries

def num_hog_features(pix_per_cell, cells_per_block, orient, imgsize=64):
    """
    Calculates the number of HOG features.
    """  
    blocks = imgsize // pix_per_cell
    return (blocks - (cells_per_block - 1)) * (blocks - (cells_per_block - 1)) *  cells_per_block * cells_per_block *  orient
    
def create_feature_row(imgname, isVehicle, pix_per_cell, cells_per_block, orient):
    """
    Computes HOG features and returns a feature row. 
    First, the image is grayscaled and the HOG function is called.
    The last entry is the label and the remaining ones describe the 
    compute HOG feature vector. The number of feature items depends on the block 
    and cell size and the number of orientations.
    """
    img = cv2.imread(imgname)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    hog_features = hog(img_gray, 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       visualise=False, 
                       feature_vector=True)
    
    _, _, _, _, hist_features = color_hist(img)
    
    feature_list = np.concatenate((hog_features, hist_features)).tolist()
    feature_list.insert(0, imgname)
    feature_list.append(1 if isVehicle else 0)
    
    return feature_list


def write_features(filename, feature_rows, header):
    df = pd.DataFrame(feature_rows, columns=header)
    if not os.path.isfile(filename):                
        df.to_csv(filename, sep='\t')
    else:
        df.to_csv(filename, sep='\t', mode='a', header=False)    

def generate_datasets():
    cars = glob.glob('training_set/vehicles/**/*.png')
    notcars = glob.glob('training_set/non-vehicles/**/*.png')
    
    pix_per_cell = 8
    cell_per_block = 2
    
    for i in range(6,13,1):
        orient = i
    
        filename = 'dataset_8_2_{}_3_32.csv'.format(orient)
    
        print('generating ', filename)
        
        num_features = calc_number_of_features(pix_per_cell, cell_per_block, orient)
        header = create_header(num_features + 3 * 32, pix_per_cell, cell_per_block, orient)
        
        print(num_features, len(header))
        
        feature_rows = []    
        counter = 0
        
        # iterate over each car image
        for car in cars:
            row = create_feature_row(car, True, pix_per_cell, cell_per_block, orient)
            feature_rows.append(row)
            
            if  counter % 500 == 0:
                print("Added 100 car feature rows. Rows total ", str(len(feature_rows)))
                write_features(filename, feature_rows, header)
                feature_rows = []
                
            counter += 1
        
        write_features(filename, feature_rows, header)
        feature_rows = []
        counter = 0
            
        # iterate over each non-car image
        for notcar in notcars:
            row = create_feature_row(notcar, False, pix_per_cell, cell_per_block, orient)
            feature_rows.append(row)
            
            if  counter % 500 == 0:
                print("Added 100 notcar feature rows. Rows total ", str(len(feature_rows)))
                write_features(filename, feature_rows, header)
                    
                feature_rows = []
                
            counter += 1
        
        write_features(filename, feature_rows, header)
        feature_rows = []

generate_datasets()
