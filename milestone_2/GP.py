#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:46:28 2018

@author: Nigel
"""

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn import neural_network
from sklearn import svm
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.metrics.classification import accuracy_score, log_loss


def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
        
# =============================================================================
validate_cmdline_args(2,'Usage: python MultinomialLogRegression.py <DATASET_PATH>')
# When 2 arguments given: python GP.py <DATAPATH>, 
# it will split the data into train:test=7:3
if len(sys.argv)==2:
    #DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
    DATASET_PATH = sys.argv[1]
    data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
    data = pd.read_csv(DATASET_PATH,names=data_features)
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[1:11]],data["eval"], train_size=0.7)
# When 3 arguments given: python GP.py <DATAPATH1> <DATAPATH2>,
# DATAPATH1: training set, DATAPATH2: test set
elif len(sys.argv)==3: 
    DATASET_PATH = sys.argv[1]
    DATASET_PATH2 = sys.argv[2]
    data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
    data1 = pd.read_csv(DATASET_PATH,names=data_features)
    data2= pd.read_csv(DATASET_PATH2,names=data_features)
    train_x = data1[data_features[1:11]]
    test_x  = data2[data_features[1:11]]
    train_y = data1["eval"]
    test_y  = data2["eval"]
# =============================================================================
 # GP with RBF Kernel
print("");
 #Kernel’s hyperparameters are optimized during fitting.
print("GP with RBF Kernel")
# Multiclass as One-vs-All
mul_gp1 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_rest').fit(train_x, train_y)
 
print('Multiclass (One-vs-All) Gaussian Process Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_gp1.predict(train_x))))
print('Multiclass (One-vs-All) Gaussian Process Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_gp1.predict(test_x))))
print("Computing CV...")
cv_gp1 = cross_val_score(mul_gp1, data[data_features[:10]], data[data_features[11]], cv=10)
#print('CV-prediction error rate :: {}'.format(cv_gp1))
#mean cv and the 95% confidence interval of the cv's estimate
print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp1.mean(), cv_gp1.std() * 2)) 

# Multiclass as One-vs-One
mul_gp2 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_one').fit(train_x, train_y)

print('Multiclass (One-vs-One) Gaussian Process Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_gp2.predict(train_x))))
print('Multiclass (One-vs-One) Gaussian Process Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_gp2.predict(test_x))))
print("Computing CV...")
cv_gp2 = cross_val_score(mul_gp2, data[data_features[1:11]], data["eval"], cv=10)
#print('CV-prediction error rate :: {}'.format(cv_gp2))
#mean cv and the 95% confidence interval of the cv's estimate
print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp2.mean(), cv_gp2.std() * 2))
# =============================================================================


# =============================================================================
 # GP with Matern Kernel
print("")
print("GP with Matern Kernel") #Kernel’s hyperparameters are optimized during fitting.
# Multiclass as One-vs-All
mul_gp1 = gaussian_process.GaussianProcessClassifier(kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5), multi_class='one_vs_rest').fit(train_x, train_y)

print('Multiclass (One-vs-All) Gaussian Process Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_gp1.predict(train_x))))
print('Multiclass (One-vs-All) Gaussian Process Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_gp1.predict(test_x))))
print("Computing CV...")
cv_gp1 = cross_val_score(mul_gp1, data[data_features[1:11]], data["eval"], cv=10)
print('CV-prediction error rate :: {}'.format(cv_gp1))
#mean cv and the 95% confidence interval of the cv's estimate
print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp1.mean(), cv_gp1.std() * 2)) 

# Multiclass as One-vs-One
mul_gp2 = gaussian_process.GaussianProcessClassifier(kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5), multi_class='one_vs_one').fit(train_x, train_y)

print('Multiclass (One-vs-One) Gaussian Process Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_gp2.predict(train_x))))
print('Multiclass (One-vs-One) Gaussian Process Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_gp2.predict(test_x))))
print("Computing CV...")
cv_gp2 = cross_val_score(mul_gp2, data[data_features[1:11]], data["eval"], cv=10)
#print('CV-prediction error rate :: {}'.format(cv_gp2))
#mean cv and the 95% confidence interval of the cv's estimate
print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp2.mean(), cv_gp2.std() * 2))
# =============================================================================

