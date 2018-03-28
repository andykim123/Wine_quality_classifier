#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tu Mar 27 15:58:01 2018

@author: J
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python clustering.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
DATASET_PATH_RED = sys.argv[1]
DATASET_PATH_WHITE = sys.argv[2]
data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]

data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_WHITE,names=data_features)
test_x_red = data_red[data_features[0:11]]
test_y_red = data_red["eval"]
test_x_white = data_white[data_features[0:11]]
test_y_white = data_white["eval"]
normalized_x_red = StandardScaler().fit_transform(test_x_red)
normalized_x_white = StandardScaler().fit_transform(test_x_white)
# X_train = np.load('data.npy')
# print(X_train)
"""
'full' (each component has its own general covariance matrix),
'tied' (all components share the same general covariance matrix),
'diag' (each component has its own diagonal covariance matrix),
'spherical' (each component has its own single variance).
"""
Ks = range(1, 11)
# km[i].means_ : # clusters x # features
# km[i].covariances_ : # clusters x # features x # features
"""red"""
for cv_type in ['full', 'tied', 'diag', 'spherical']:
    km = [GaussianMixture(n_components=i, covariance_type=cv_type).fit(normalized_x_red) for i in Ks]
    labels = [km[i].predict(normalized_x_red) for i in range(len(km))]
    scores = [accuracy_score(test_y_red, labels[i]) for i in range(len(km))]
    print(scores)
    plt.plot(Ks,scores)
    plt.ylabel('accuracy')
    plt.xlabel('# clusters')
    plt.ylim(0,1)
    plt.title('RED using covariance type: '+cv_type)
    plt.show()
# ===================================================================================================
"""white"""
for cv_type in ['full', 'tied', 'diag', 'spherical']:
    km = [GaussianMixture(n_components=i, covariance_type=cv_type).fit(normalized_x_white) for i in Ks]
    labels = [km[i].predict(normalized_x_white) for i in range(len(km))]
    scores = [accuracy_score(test_y_white, labels[i]) for i in range(len(km))]
    print(scores)
    plt.plot(Ks,scores)
    plt.ylabel('accuracy')
    plt.xlabel('# clusters')
    plt.ylim(0,1)
    plt.title('RED using covariance type: '+cv_type)
    plt.show()
