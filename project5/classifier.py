# -*- coding: utf-8 -*-
"""
Created on Sat May 13 08:33:27 2017

@author: rudi
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from sklearn import metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# taken from: 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train_svc(X, y, tune=True):
    print("training svm...")
    
    if tune == True:
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svr = svm.SVC()
        clf = GridSearchCV(svr, parameters)
    else:
        clf = svm.SVC( kernel='linear', C=1)
        
    clf.fit(X, y)
    return clf

def calc_metrics(y_test, y_pred):    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1score, cnf_matrix

def evaluate(classifier, X_test, y_test):
    print("evaluating svm...")
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy, precision, recall, f1score, cnf_matrix = calc_metrics(y_test, y_pred)
    return accuracy, precision, recall, f1score, cnf_matrix

def dump(clf, scaler, filename):
    # store tuned model
    print('dumping model and scaler')
    joblib.dump({'clf' : clf, 'scaler' : scaler}, filename)
    
def load(filename):
    # load model
    return joblib.load( filename )

def train_models():
    #filenames = ['dataset_8_2_6_3_32.tsv', 'dataset_8_2_7_3_32.tsv', 'dataset_8_2_8_3_32.tsv', 'dataset_8_2_9_3_32.tsv']
    filenames = ['dataset_8_2_9_3_32.tsv']
    
    classifiers = []
    
    for filename in filenames:
        print("train svm model with ", filename)
        
        dataset = pd.read_csv(filename, sep='\t')
        dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
        
        # remove the imagename column which is not used for training
        popped = dataset.pop('imagename')
        num_features = len(dataset.columns)
        
        shuffled_dataset = shuffle(dataset.values)        
        shuffled_dataset = shuffle(shuffled_dataset)
        
        X = shuffled_dataset[:,0:num_features-1].astype(np.float64)        
        y = np.concatenate(shuffled_dataset[:,num_features-1:num_features].reshape(1,-1))
        
        print(X.shape, ' ', y.shape)
        
        # Normalize features
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        #X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
        
        # train a support vector machine
        svm = train_svc(X_train, y_train, tune=False)
        
        # eval model
        # calculate some metrics
        accuracy, precision, recall, f1score, cnf_matrix = evaluate(svm, X_test, y_test)
        
        print('Accuracy ', accuracy)
        print('Precision ', precision)
        print('Recall ', recall)
        print('F1 score ', f1score)
        
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['NoCar', 'Car'], title='Confusion matrix, without normalization')
        
        
        base = os.path.basename(filename)
        name = os.path.splitext(base)
        # serialize model
        dump(svm, X_scaler, "svm_{}.pkl".format(name[0]))
        
        classifiers.append(svm)
    return classifiers


#train_models()
