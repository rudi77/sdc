# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:27:38 2017

@author: rudi
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D