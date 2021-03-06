# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:43:26 2018

@author: Nigel

This file includes the code for CSE517A(spring 2018) : Milestone 1(Linear Classifier)
"""

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python LogisticRegression.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
DATASET_PATH_RED = sys.argv[1]
DATASET_PATH_WHITE = sys.argv[2]

#DATASET_PATH_RED = "/Users/Nigel/Desktop/Wash U/2018 Junior Spring/CSE 517a/Milestone Projects Local Repo/milestone_1/winequality-red.csv"
#DATASET_PATH_WHITE = "/Users/Nigel/Desktop/Wash U/2018 Junior Spring/CSE 517a/Milestone Projects Local Repo/milestone_1/winequality-red.csv"
data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_RED,names=data_features)

                                    ## Red Wine ##

train_x, test_x, train_y, test_y = train_test_split(data_red[data_features[:10]],data_red[data_features[11]], train_size=0.7)

print('-------------Red Wine Evaluation-------------')
# Multiclass Logistic Regression
mul_lr = linear_model.LogisticRegression(fit_intercept=True, multi_class='multinomial', solver = 'newton-cg')
mul_lr_fit = mul_lr.fit(train_x, train_y)

print('Multiclass Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr_fit.predict(train_x))))
print('Multiclass Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr_fit.predict(test_x))))
print('CV-prediction error rate :: {}'.format(cross_val_score(mul_lr, data_red[data_features[:10]], data_red[data_features[11]], cv=10)))

train_y[train_y<5] = 0
train_y[train_y>=5] = 1
test_y[test_y<5] = 0
test_y[test_y>=5] = 1
cv_x = data_red[data_features[:10]]
cv_y = data_red[data_features[11]]
cv_y[cv_y<5] = 0
cv_y[cv_y>=5] = 1

# Binary Classification Using Logistic Regression
lr = linear_model.LogisticRegression(fit_intercept=True)
lr_fit = lr.fit(train_x,train_y)

print('Binary Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, lr_fit.predict(train_x))))
print('Binary Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, lr_fit.predict(test_x))))
print('Binary CV-prediction error rate :: {}'.format(cross_val_score(lr, cv_x, cv_y, cv=10)))

                                    ## White Wine ##

train_x_w, test_x_w, train_y_w, test_y_w = train_test_split(data_white[data_features[:10]],data_white[data_features[11]], train_size=0.7)

print('-------------White Wine Evaluation-------------')
# Multiclass Logistic Regression
mul_lr = linear_model.LogisticRegression(fit_intercept=True, multi_class='multinomial', solver = 'newton-cg')
mul_lr_fit = mul_lr.fit(train_x_w, train_y_w)

print('Multiclass Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y_w, mul_lr_fit.predict(train_x_w))))
print('Multiclass Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y_w, mul_lr_fit.predict(test_x_w))))
print('CV-prediction error rate :: {}'.format(cross_val_score(mul_lr, data_white[data_features[:10]], data_white[data_features[11]], cv=10)))

train_y_w[train_y_w<5] = 0
train_y_w[train_y_w>=5] = 1
test_y_w[test_y_w<5] = 0
test_y_w[test_y_w>=5] = 1
cv_x = data_white[data_features[:10]]
cv_y = data_white[data_features[11]]
cv_y[cv_y<5] = 0
cv_y[cv_y>=5] = 1

# Binary Classification Using Logistic Regression
lr = linear_model.LogisticRegression(fit_intercept=True)
lr_fit = lr.fit(train_x_w, train_y_w)

print('Binary Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y_w, lr_fit.predict(train_x_w))))
print('Binary Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y_w, lr_fit.predict(test_x_w))))
print('Binary CV-prediction error rate :: {}'.format(cross_val_score(lr, cv_x, cv_y, cv=10)))
