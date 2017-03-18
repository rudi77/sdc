# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:26:07 2017

@author: rudi
"""
import os
import pickle

def persist(filename, features, labels, path='traffic-signs-data-augmented'):
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
    
    file = os.path.join(path, filename)
    
    try:
        with open(file, mode='rb') as pfile:
            return pickle.load(pfile)
    except Exception as e:
        print('Unable to open file', file, ':', e)
        raise
