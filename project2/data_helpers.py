# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:56:21 2017

@author: rudi
"""
import os
import pandas as pd
import cv2
import pickle

#import data_transformations as dt
#import data_plotting as dplt

def persist(filename, features, labels, path='traffic-signs-data-augmented'):
    """
    Persists images and labels in a pickle file
    """
    assert(len(features) == len(labels))
    
    file = os.path.join(path, filename)
    
    # Save the data for easy access
    if not os.path.isfile(file):
        print('Saving data to pickle file...')
        try:
            with open(file, 'wb') as pfile:
                pickle.dump(
                    {
                        'features': features,
                        'labels': labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    
    print('Data cached in pickle file.')
    
def load(filename, path='traffic-signs-data-augmented'):
    """
    Loads images and labels from a pickle file.
    """
    file = os.path.join(path, filename)
    
    try:
        with open(file, mode='rb') as pfile:
            return pickle.load(pfile)
    except Exception as e:
        print('Unable to open file', file, ':', e)
        raise

def create_trafficsign_map(csvfile):
    """
    Reads the csv file that contains signnames with their classids
    and returns a dictionary where the Key is the classId and the Value
    is the name of the traffic sign
    """
    df = pd.read_csv(csvfile)
    
    trafficsignmap = { row['ClassId'] : row['SignName'] for index, row in df.iterrows() }
    
    return trafficsignmap

def create_testset(rootpath):
    """
    Creates a test set used to validate a trained network. There exists a root directory
    and within this directory there exists a directory for each traffic sign class using
    the ClassId as directory name. We iterate over each directory, read all images
    label them and return an array of all images and and a corresponding label array.
    """    
    # TODO: check if rootpath exists
    
    dirs = os.listdir(rootpath)
    imagedirs = []
    
    for item in dirs:
        possibledir = os.path.join(rootpath, item)
        
        if os.path.isdir(possibledir):
            imagedirs.append(possibledir)
    
    images = []
    labels = []

    for dir in imagedirs:
        #if dir not empty
        files =  os.listdir(dir)
        for file in files:
            # TODO: also check if filename can be casted to an integer representing
            # a certain traffic sign class
            imagepath = os.path.join(dir, file)
            
            if os.path.isfile(imagepath): 
                image = cv2.imread(imagepath)
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resizedimg = cv2.resize(imageRGB, (32, 32))
                images.append( resizedimg )
                
                path, foldername = os.path.split(dir)
                labels.append(int(foldername))
                
    return images, labels

rootpath = "C:\\Users\\rudi\\Documents\\Udacity\\SelfDrivingCar\\sdc\\project2\\examples\\MyGermanTrafficSigns"
X_test, y_test = create_testset(rootpath)

#dplt.set_plotsize(40,30)
#dplt.showimages(X_test, y_test, counts=len(y_test), rows=6, cols=8, isRandom=True)
#print(labels)

#signnames = "C:\\Users\\rudi\\Documents\\Udacity\\SelfDrivingCar\\sdc\\project2\\signnames.csv"
#tsmap = create_trafficsign_map(signnames)
#print(tsmap)
