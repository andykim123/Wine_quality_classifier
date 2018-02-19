#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:45:18 2018

@author: dohoonkim
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

DATASET_PATH = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)
train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)


lr = linear_model.LogisticRegression()
lr.fit(train_x, train_y)
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

print('Multinomial Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr.predict(train_x))))
print('Multinomial Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr.predict(test_x))))

