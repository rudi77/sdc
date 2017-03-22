# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:54:30 2017

@author: rudi
"""

import matplotlib.pyplot as plt
import numpy as np
import data_helpers as dh

plt.rcParams['figure.figsize'] = (40, 30)

def set_plotsize(x,y):
    plt.rcParams['figure.figsize'] = (x, y)
    
def get_plotsize():
    return plt.rcParams['figure.figsize'] 

def showimages(images, labels, counts=10, rows=2, cols=5, isRandom=True, isGray=False):
    assert(len(images) == len(labels))
    
    trafficsignmap = dh.create_trafficsign_map('signnames.csv')
    
    fig = plt.figure()
    for i in range(counts):
        if isRandom:
            idx = np.random.randint(len(images))
        else:
            idx = i
        image = images[idx]
        ax = fig.add_subplot(rows, cols, i+1)
        if not isGray: 
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray_r")
        ax.set_title('%s' % (trafficsignmap[labels[idx]]))
        
def histogram(samples, classes):
    # distribution of examples per class displayed as a histogram
    plt.rcParams['figure.figsize'] = (8, 6)
    n, bins, patches = plt.hist(samples, classes, facecolor='green', histtype='bar', rwidth=0.8, alpha=1)
    plt.grid(True)
     
    
def top5_barchart(probabilities, indices):
    plt.rcParams['figure.figsize'] = (8, 6)
    trafficsignmap = dh.create_trafficsign_map('signnames.csv')
    trafficsigns = [trafficsignmap[idx] for idx in indices]
    
    plt.figure(1)
    plt.barh(indices,probabilities, align='center', color='g')
    plt.yticks(indices, trafficsigns)
    plt.xlabel('Probabilities')
    plt.grid(True)


def show_top5(images, probs, indices):
    assert(len(images) == len(probs) == len(indices))

    trafficsignmap = dh.create_trafficsign_map('signnames.csv')
    plt.rcParams['figure.figsize'] = (16, 4)
    
    pos = np.arange(5)
    
    
    for i in range(len(images)):
        img  = images[i]
        prob = probs[i]
        idxs = indices[i]
        
        trafficsigns = [trafficsignmap[idx] for idx in idxs]
        
        fig = plt.figure()
        ax_ts = fig.add_subplot(1,2,1)
        ax_ts.imshow(img, cmap='gray_r')
        
        ax_p = fig.add_subplot(1, 2, 2)       
        ax_p.barh(pos, prob, align='center', color='g')
        ax_p.set_yticks(pos)
        ax_p.set_yticklabels(trafficsigns)
        ax_p.set_xlabel('Probabilities')
        ax_p.grid(True)  
        #plt.show()
        
        
import data_helpers as dh
myexamplespath = ".\examples\MyGermanTrafficSigns"
X_test, y_test = dh.create_testset(myexamplespath)
probs = [[0.5, 0.2, 0.1, 0.05, 0.025], [0.5, 0.2, 0.1, 0.05, 0.025], [0.5, 0.2, 0.1, 0.05, 0.025]]
indices = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
show_top5(X_test[0:3], probs, indices)