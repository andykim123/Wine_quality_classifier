#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:45:18 2018

@author: dohoonkim
"""

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python MultinomialLogRegression.py <RUN INFILE BOOLEAN> <DATASET_PATH>')

run_infile = False

if(sys.argv[1]=="true" or sys.argv[1]=="True"):
    run_infile = True

data_features = ["f1","f2","f3","f4","f5","f6","f7","f8","score"]

if not run_infile:
    DATASET_PATH = sys.argv[2]
    data = pd.read_csv(DATASET_PATH,names=data_features)
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:7]],data[data_features[8]], train_size=0.7)
    # lr = linear_model.LogisticRegression()
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    # lr.fit(train_x, train_y)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)
    mul_lr_fit = mul_lr.fit(train_x, train_y)
    print('Multinomial Logistic regression Multionmial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr_fit.predict(train_x))))
    print('Multinomial Logistic regression Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr_fit.predict(test_x))))
    print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(mul_lr, data[data_features[:7]], data[data_features[8]], cv=10)))
    train_y[train_y<5] = 0
    train_y[train_y>=5] = 1
    test_y[test_y<5] = 0
    test_y[test_y>=5] = 1
    cv_x = data_red[data_features[:7]]
    cv_y = data_red[data_features[8]]
    cv_y[cv_y<5] = 0
    cv_y[cv_y>=5] = 1
    mul_bi_fit = mul_lr.fit(train_x, train_y)
    print('Multinomial Logistic regression Binary Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_bi_fit.predict(train_x))))
    print('Multinomial Logistic regression Binary Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_bi_fit.predict(test_x))))
    print('Binary CV-prediction error rate :: {}'.format(cross_val_score(mul_lr, cv_x, cv_y, cv=10)))
else:
    DATASET_PATH_TRAIN = sys.argv[2]
    DATASET_PATH_TEST = sys.argv[3]
    train = pd.read_csv(DATASET_PATH_TRAIN)
    test = pd.read_csv(DATASET_PATH_TEST)
    train_x = train[data_features[0:7]]
    train_y = train["score"]
    test_x = test[data_features[0:7]]
    test_y = test["score"]
    # train_x, dummy_x, train_y, dummy_y = train_test_split(train[data_features[:7]],train[data_features[8]], train_size=1)
    # test_x, dummy_x, test_y, dummy_y = train_test_split(test[data_features[:7]],test[data_features[8]], train_size=1)
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mul_lr_fit = mul_lr.fit(train_x, train_y)
    print(metrics.accuracy_score(test_y, mul_lr_fit.predict(test_x)))
