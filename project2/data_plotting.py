# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:54:30 2017

@author: rudi
"""

import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import data_helpers as dh
=======
>>>>>>> 9bd74fa7d2f914731adfd077ad75f5c29f104b1d

plt.rcParams['figure.figsize'] = (40, 30)

def set_plotsize(x,y):
    plt.rcParams['figure.figsize'] = (x, y)
    
def get_plotsize():
    return plt.rcParams['figure.figsize'] 

def showimages(images, labels, counts=10, rows=2, cols=5, isRandom=True):
    assert(len(images) == len(labels))
    
<<<<<<< HEAD
    trafficsignmap = dh.create_trafficsign_map('signnames.csv')
    
=======
>>>>>>> 9bd74fa7d2f914731adfd077ad75f5c29f104b1d
    fig = plt.figure()
    for i in range(counts):
        if isRandom:
            idx = np.random.randint(len(images))
        else:
            idx = i
        image = images[idx]
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image)
<<<<<<< HEAD
        ax.set_title('%s' % (trafficsignmap[labels[idx]]))
=======
        ax.set_title('Class: %d' % (labels[idx]))
>>>>>>> 9bd74fa7d2f914731adfd077ad75f5c29f104b1d
        
def histogram(samples, classes):
    # distribution of examples per class displayed as a histogram
    plt.rcParams['figure.figsize'] = (8, 6)
    n, bins, patches = plt.hist(samples, classes, facecolor='green', histtype='bar', rwidth=0.8, alpha=1)
    plt.grid(True)
        