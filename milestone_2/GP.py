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
validate_cmdline_args(3,'Usage: python MultinomialLogRegression.py <DATASET_PATH> OPTIONAL(<TEST_DATA_PATH>) <RUN INFILE BOOLEAN>')
# run_infile_boolean is a boolean that checks whether a particular run is done within other python file or not
# if it is true, it indicates that the run is done within other python file run. If false, it is done in command line.
# When 3 arguments given: python GP.py <DATAPATH> <RUN INFILE BOOLEAN>, 
# it will split the data into train:test=7:3

run_infile = False

if len(sys.argv)==3:
    #DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
    DATASET_PATH = sys.argv[1]
    data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
    data1 = pd.read_csv(DATASET_PATH,names=data_features)
    train_x, test_x, train_y, test_y = train_test_split(data1[data_features[0:11]],data1["eval"], train_size=0.7)
    if(sys.argv[2]=="true" or sys.argv[2]=="True"):
        run_infile = True
    if not run_infile:
        print("\nSplitting... '%s' into Training set and Test set...\n" % DATASET_PATH[DATASET_PATH.rfind("/")+1: ])
# When 3 arguments given: python GP.py <DATAPATH1> <DATAPATH2>,
# DATAPATH1: training set, DATAPATH2: test set
elif len(sys.argv)==4: 
    DATASET_PATH1 = sys.argv[1]
    DATASET_PATH2 = sys.argv[2]
    run_infile = sys.argv[3]
    data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
    data1 = pd.read_csv(DATASET_PATH1,names=data_features)
    data2= pd.read_csv(DATASET_PATH2,names=data_features)
    train_x = data1[data_features[1:11]]
    test_x  = data2[data_features[1:11]]
    train_y = data1["eval"]
    test_y  = data2["eval"]
    if(sys.argv[3]=="true" or sys.argv[3]=="True"):
        run_infile = True
    if not run_infile:
        print("Training... '%s'\n" % DATASET_PATH1[DATASET_PATH1.rfind("/")+1: ])
        print("Testing... '%s'\n" % DATASET_PATH2[DATASET_PATH2.rfind("/")+1: ])
    
if not run_infile:
    # =============================================================================
    # GP with RBF Kernel
    #Kernel’s hyperparameters are optimized during fitting.
    print("\nGP with RBF Kernel\n")
    
    # Multiclass as One-vs-All
    t1 = time.time()
    #kernel=1.0 * RBF(length_scale=1.0)
    mul_gp1 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_rest').fit(train_x, train_y)
    #print(mul_gp1.predict(train_x))
    elapsed = time.time() - t1
    print('Multiclass (1-vs-All) computation time :: %.3f\n' % (elapsed))
    print('Multiclass (1-vs-All) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp1.predict(train_x))))
    print('Multiclass (1-vs-All) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp1.predict(test_x))))
    print("Negative Log Likelihood: %.3f\n" % (mul_gp1.log_marginal_likelihood(theta=None)))
    
    print("Computing 10-fold CV...\n")
    tcv = time.time()
    cv_gp1 = cross_val_score(mul_gp1, data1[data_features[0:11]], data1["eval"], cv=10)
    elapsed = time.time() - tcv
    print('CV computation time :: %.3f\n' % (elapsed))
    ##print('CV-prediction error rate :: {}'.format(cv_gp1))
    ##mean cv and the 95% confidence interval of the cv's estimate
    print("Accuracy(Mean CV): %0.2f (+/- %0.2f)\n" % (cv_gp1.mean(), cv_gp1.std() * 2))
    print('---------------------------------------------')  
    
    # Multiclass as One-vs-One
    t2 = time.time()
    #kernel=1.0 * RBF(length_scale=1.0)
    mul_gp2 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_one').fit(train_x, train_y)
    elapsed = time.time() - t2
    print('Multiclass (1-vs-1) computation time :: %.3f\n' % (elapsed))
    print('Multiclass (1-vs-1) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp2.predict(train_x))))
    print('Multiclass (1-vs-1) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp2.predict(test_x))))
    print("Negative Log Likelihood: %.3f\n" % (mul_gp1.log_marginal_likelihood(theta=None)))
    
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
     # GP with Matern Kernel
    print("")
    print("\nGP with Matern Kernel\n") 
    
    # Multiclass as One-vs-All
    t1 = time.time()
    mul_gp1 = gaussian_process.GaussianProcessClassifier(kernel=1.0*Matern(length_scale=1.0, nu=1.5), multi_class='one_vs_rest').fit(train_x, train_y)
    elapsed = time.time() - t1
    print('Multiclass (1-vs-All) computation time :: %.3f\n' % (elapsed))
    
    print('Multiclass (1-vs-All) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp1.predict(train_x))))
    print('Multiclass (1-vs-All) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp1.predict(test_x))))
    print("Negative Log Likelihood: %.3f\n" % (mul_gp1.log_marginal_likelihood(theta=[math.log(1), math.log(1)])))
    
    print("Computing 10-fold CV...\n")
    tcv = time.time()
    cv_gp1 = cross_val_score(mul_gp1, data1[data_features[0:11]], data1["eval"], cv=10)
    elapsed = time.time() - tcv
    print('CV computation time :: %.3f\n' % (elapsed))
    #print('CV-prediction error rate :: {}'.format(cv_gp1))
    #mean cv and the 95% confidence interval of the cv's estimate
    print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp1.mean(), cv_gp1.std() * 2))
    print('---------------------------------------------') 
    
    # Multiclass as One-vs-One
    t2 = time.time()
    mul_gp2 = gaussian_process.GaussianProcessClassifier(kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                            nu=1.5), multi_class='one_vs_one').fit(train_x, train_y)
    elapsed = time.time() - t2
    print('Multiclass (1-vs-1) computation time :: %.3f\n' % (elapsed))
    
    print('Multiclass (1-vs-1) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp2.predict(train_x))))
    print('Multiclass (1-vs-1) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp2.predict(test_x))))
    print("Negative Log Likelihood: %.3f\n" % (mul_gp1.log_marginal_likelihood(theta=[math.log(1), math.log(1)])))
    
    print("Computing 10-fold CV...")
    tcv = time.time()
    cv_gp2 = cross_val_score(mul_gp2, data1[data_features[0:11]], data1["eval"], cv=10)
    elapsed = time.time() - tcv
    print('CV computation time :: %.3f\n' % (elapsed))
    #print('CV-prediction error rate :: {}'.format(cv_gp2))
    #mean cv and the 95% confidence interval of the cv's estimate
    print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp2.mean(), cv_gp2.std() * 2))
    print('---------------------------------------------') 
    #print("Log-loss: %.3f" % (log_loss(train_y, mul_gp2.predict_proba(train_x)[:, 1])))
    
    # =============================================================================
else:
    #if the run in done within modelEvaluation.py, we just return cross_val_score result, which is a list of 10 different float-type accuracies
    mul_gp = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_rest').fit(train_x, train_y)
    cv = cross_val_score(mul_gp, data1[data_features[0:11]], data1["eval"], cv=10)
    print(cross_val_score(mul_gp, data1[data_features[0:11]], data1["eval"], cv=10))
    print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))