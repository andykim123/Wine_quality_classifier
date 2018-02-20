# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:43:26 2018

@author: Nigel
"""

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

#def validate_cmdline_args(nargs, msg):
#    if len(sys.argv) < nargs:
#        print(msg)
#        sys.exit(1)
#validate_cmdline_args(2,'Usage: python MultinomialLogRegression.py <DATASET_PATH>')
##DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
#DATASET_PATH = sys.argv[1]


DATASET_PATH = "/Users/Nigel/Desktop/Wash U/2018 Junior Spring/CSE 517a/Milestone Projects Local Repo/milestone_1/winequality-red.csv"
data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)

train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)

train_y[train_y<5] = 0
train_y[train_y>=5] = 1
test_y[test_y<5] = 0
test_y[test_y>=5] = 1

mul_lr = linear_model.LogisticRegression(fit_intercept=True).fit(train_x, train_y)

print('Binary Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr.predict(train_x))))
print('Binary Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr.predict(test_x))))