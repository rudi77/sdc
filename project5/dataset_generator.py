# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:18:03 2017

@author: rudi
"""
import os
import time
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

def to_colorspace(img, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'Gray':
            image_converted = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: 
        image_converted = np.copy(img)
        
    return image_converted
        
    

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, size=(32, 32), asvec=True):    
    return cv2.resize(img, size).ravel() if asvec == True else cv2.resize(img, size)

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), histonly=True):
    
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1]) / 2
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    if histonly:
        return hist_features
    else:
        # Return the individual histograms, bin_centers and feature vector
        return rhist, ghist, bhist, bin_centers, hist_features
    
def hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):    
    if vis == True:
        features, hog_img = hog(img, 
                                orientations=orient, 
                                pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                cells_per_block=(cell_per_block, cell_per_block),
                                visualise=vis, 
                                feature_vector=feature_vec)
        return features, hog_img
    else:      
        return hog(img, 
                   orientations=orient, 
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), 
                   visualise=vis, 
                   feature_vector=feature_vec)    

def calc_number_of_features(pix_per_cell, cells_per_block, orient, imgsize=64):
    """
    Calculates the number of HOG features.
    """  
    blocks = imgsize // pix_per_cell
    return (blocks - (cells_per_block - 1)) * (blocks - (cells_per_block - 1)) *  cells_per_block * cells_per_block *  orient

def create_header(num_features, pix_per_cell, cells_per_block, orient):
    """
    Creates the header for our dataset.
    """
    header_entries = [i+1 for i in range(num_features)]    
    header_entries.insert(0,'imagename')
    header_entries.append('label')
    
    return header_entries

def num_hog_features(pix_per_cell, cells_per_block, orient, imgsize=64):
    """
    Calculates the number of HOG features.
    """  
    blocks = imgsize // pix_per_cell
    return (blocks - (cells_per_block - 1)) * (blocks - (cells_per_block - 1)) *  cells_per_block * cells_per_block *  orient

def create_feature_row(img, pix_per_cell=8, cells_per_block=2, orient=9, color_bins=16, spat_size=(16,16), label = None, imgname = None, useAll=True, useCols=True):
    """
    Computes a feature descriptor for the provided image. This feature vector contains different feature
    types. HoG, Color Histograms and Spatial binnings.
    The feature descriptor is returned as a single list of feature entries.
    """

    features = []
    img_converted = to_colorspace(img, 'YCrCb')
    if useAll == True:
        for i in range(3):
            hog = hog_features(img_converted[:,:,i], orient, pix_per_cell, cells_per_block, feature_vec=True)    
            features.extend(hog)
    else:
        img_gray = to_colorspace(img, 'Gray')
        hog = hog_features(img_gray, orient, pix_per_cell, cells_per_block, feature_vec=True)    
        features.extend(hog)
    
    if useCols == True:
        col_features = color_hist((img_converted * 255).astype(np.uint8), nbins=color_bins)
        spat_features = bin_spatial(img_converted, size=spat_size)
        features.extend(col_features)
        features.extend(spat_features)
    
    if label is None:
        return np.concatenate(features).tolist()
    else:
        # This branch is used when a trainingset is generated. Here, the image path and the label
        # is added to the vector
        assert(imgname is not None)        
        return np.concatenate(([imgname], features, [label])).tolist()


def generate_datasets():
    
    def write_features(filename, feature_rows, header):
        df = pd.DataFrame(feature_rows, columns=header)
        if not os.path.isfile(filename):                
            df.to_csv(filename, sep='\t')
        else:
            df.to_csv(filename, sep='\t', mode='a', header=False)    
    
    cars = glob.glob('training_set/vehicles/**/*.png')
    notcars = glob.glob('training_set/non-vehicles/**/*.png')
    
    CAR = 1
    NOTCAR = 0
    
    pix_per_cell = 8
    cell_per_block = 2
    
    # Generates datasets for 6,7,8 and 9 HOG orientations
    for i in range(9,10,1):
        orient = i
    
        filename = 'dataset_8_2_{}_3_16_3_16_16_all.tsv'.format(orient)
        
        print('generating ', filename)
        
        num_features = calc_number_of_features(pix_per_cell, cell_per_block, orient)
        
        # 3*16 color bins, 16*16*3 spatial bins
        header = create_header(num_features*3 + 3*16 + 16*16*3, pix_per_cell, cell_per_block, orient)
        #header = create_header(num_features, pix_per_cell, cell_per_block, orient)

        feature_rows = []    
        counter = 0
        tStart = time.time()
        # iterate over each car image
        for car in cars:
            img = mpimg.imread(car)
            row = create_feature_row(img, pix_per_cell, cell_per_block, orient, label=CAR, imgname=car, useAll=True, useCols=True)
            feature_rows.append(row)
            
            if  counter % 500 == 0:
                print("Writing car feature rows. Rows total ", str(len(feature_rows)))
                write_features(filename, feature_rows, header)
                feature_rows = []
                
            counter += 1
        
        if len(feature_rows):
            write_features(filename, feature_rows, header)
            feature_rows = []
        counter = 0
            
        # iterate over each non-car image
        for notcar in notcars:
            img = mpimg.imread(notcar)
            row = create_feature_row(img, pix_per_cell, cell_per_block, orient, label=NOTCAR, imgname=notcar, useAll=True, useCols=True)
            feature_rows.append(row)
            
            if  counter % 500 == 0:
                print("Writing notcar feature rows. Rows total ", str(len(feature_rows)))
                write_features(filename, feature_rows, header)
                    
                feature_rows = []
                
            counter += 1
        
        write_features(filename, feature_rows, header)
        feature_rows = []
        
        tEnd = time.time()
        print("dataset generation in {} [s]".format(round(tEnd-tStart, 5)))

#generate_datasets()
