#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:46:28 2018
@author: Nigel
"""

import pandas as pd
import numpy as np
import sys
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import time as time
import math as math


def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
        
# =============================================================================
validate_cmdline_args(3,'Usage: python GPeval.py <RUN INFILE BOOLEAN> <DATASET_PATH>')
# run_infile_boolean is a boolean that checks whether a particular run is done within other python file or not
# if it is true, it indicates that the run is done within other python file run. If false, it is done in command line.
# When 3 arguments given: python GP.py <DATAPATH> <RUN INFILE BOOLEAN>, 
# it will split the data into train:test=7:3

run_infile = False
if(sys.argv[1]=="true" or sys.argv[1]=="True"):
    run_infile = True
#if len(sys.argv)==3:
#    #DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
#    DATASET_PATH = sys.argv[2]
#   
#    if(sys.argv[1]=="true" or sys.argv[1]=="True"):
#        run_infile = True
#    if not run_infile:
#        print("\nSplitting... '%s' into Training set and Test set...\n" % DATASET_PATH[DATASET_PATH.rfind("/")+1: ])

if not run_infile:
    # =============================================================================
    # GP with RBF Kernel
    #Kernel’s hyperparameters are optimized during fitting. 
    DATASET_PATH = sys.argv[2]
    data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
    data1 = pd.read_csv(DATASET_PATH,names=data_features)
    train_x, test_x, train_y, test_y = train_test_split(data1[data_features[0:11]],data1["eval"], train_size=0.7)
    print("\nGP with RBF Kernel\n")
    
    # Multiclass as One-vs-One
    t2 = time.time()
    #kernel=1.0 * RBF(length_scale=1.0)
    mul_gp2 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_one').fit(train_x, train_y)
    elapsed = time.time() - t2
    print('Multiclass (1-vs-1) computation time :: %.3f\n' % (elapsed))
    print('Multiclass (1-vs-1) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp2.predict(train_x))))
    print('Multiclass (1-vs-1) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp2.predict(test_x))))
    print("Negative Log Likelihood: %.3f\n" % (mul_gp2.log_marginal_likelihood(theta=None)))
    
    print("Computing 10-fold CV...\n")
    tcv = time.time()
    cv_gp2 = cross_val_score(mul_gp2, data1[data_features[0:11]], data1["eval"], cv=10)
    elapsed = time.time() - tcv
    print('CV computation time :: %.3f\n' % (elapsed))
    #print('CV-prediction error rate :: {}'.format(cv_gp2))
    #mean cv and the 95% confidence interval of the cv's estimate
    print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp2.mean(), cv_gp2.std() * 2))
    print('---------------------------------------------') 
    # =============================================================================
else:
    #if the run in done within modelEvaluation.py, we just return cross_val_score result, which is a list of 10 different float-type accuracies
    DATASET_PATH_TRAIN = sys.argv[2]
    DATASET_PATH_TEST = sys.argv[3]
    data_features = ["f1","f2","f3","f4","f5","f6","f7","f8","score"]

    train = pd.read_csv(DATASET_PATH_TRAIN)
    test = pd.read_csv(DATASET_PATH_TEST)
    train_x = train[data_features[0:7]]
    train_y = train["score"]
    test_x = test[data_features[0:7]]
    test_y = test["score"]
    
    #if the run in done within modelEvaluation.py, we just return cross_val_score result, which is a list of 10 different float-type accuracies
    mul_gp = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_one').fit(train_x, train_y)
#    print(cross_val_score(mul_gp, data1[data_features[0:11]], data1["eval"], cv=10))
    print(metrics.accuracy_score(test_y, mul_gp.predict(test_x)))