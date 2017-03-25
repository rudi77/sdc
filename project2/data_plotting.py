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

def showimages(images, labels, counts=10, rows=2, cols=5, isRandom=True, isGray=False, name='', labelconverter=None):
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
            
        if labelconverter == None:
            ax.set_title('%s' % (labels[idx]))           
        else:
             #ax.set_title('%s' % (trafficsignmap[labels[idx]]))
            ax.set_title('%s' % (labelconverter(labels[idx])))
            
        
    if name and not name.isspace():
        fig.savefig(name)

def histogram_traffic_signs(trafficsigns, xlabels, titles=["Training", "Validation", "Testing"], name=''):
    colors = ['g', 'b', 'r']
    fig = plt.figure()
    indices = np.arange(len(xlabels))
    bar_width = 0.8
    
    for i in range(len(trafficsigns)):
        ax = fig.add_subplot(3,1,i+1)
        ax.set_title(titles[i])
        ax.set_xticks(indices)
        ax.set_xticklabels(xlabels)                
        ax.bar(indices, trafficsigns[i], width=bar_width, align='center', color=colors[i])
        
    if name and not name.isspace():
        fig.savefig(name)
        
def histogram(samples, xlabels, name=''):
    # distribution of examples per class displayed as a histogram
    plt.rcParams['figure.figsize'] = (8, 6)
    n, bins, patches = plt.hist(samples, len(xlabels), facecolor='green', histtype='bar', rwidth=0.8, alpha=1)
    plt.grid(True)
    plt.xticklabels(xlabels)
    
    if name and not name.isspace():
        plt.savefig(name)
     
    
def barchart(probabilities, indices, title, name=''):

    trafficsignmap = dh.create_trafficsign_map('signnames.csv')
    trafficsigns = [trafficsignmap[idx] for idx in indices]
    
    plt.figure(1)
    plt.barh(indices,probabilities, align='center', color='g')
    plt.yticks(indices, trafficsigns)
    plt.title(title)
    plt.grid(True)
    
    if name and not name.isspace():
        plt.savefig(name)
         
def show_top5(images, probs, indices, name=''):
    assert(len(images) == len(probs) == len(indices))

    trafficsignmap = dh.create_trafficsign_map('signnames.csv')    
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

        if name and not name.isspace():
            plt.savefig(name + '_' + i)
        #plt.show()
        

# samples = [[10,8,6],[4,2,1]]
# classes = [0,1,2]
# histogram_traffic_signs(samples, classes)