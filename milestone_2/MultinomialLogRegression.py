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
validate_cmdline_args(3,'Usage: python MultinomialLogRegression.py <DATASET_PATH> <RUN INFILE BOOLEAN>')

#DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
DATASET_PATH = sys.argv[1]
data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"] #12
data = pd.read_csv(DATASET_PATH,names=data_features)
train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)

lr = linear_model.LogisticRegression()
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
lr.fit(train_x, train_y)
mul_lr_fit = mul_lr.fit(train_x, train_y)

run_infile = False

if(sys.argv[2]=="true" or sys.argv[2]=="True"):
    run_infile = True

if not run_infile:
    print('Multinomial Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr_fit.predict(train_x))))
    print('Multinomial Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr_fit.predict(test_x))))
    print('CV-prediction error rate :: {}'.format(cross_val_score(lr, data[data_features[:10]], data[data_features[11]], cv=10)))
else:
    x_fit = preprocessing.StandardScaler().fit(data[data_features[0:11]])
    data_x = x_fit.transform(data[data_features[0:11]])
    print(cross_val_score(mul_lr, data_x, data["eval"], cv=10))