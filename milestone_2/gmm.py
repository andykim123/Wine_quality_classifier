#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tu Mar 27 15:58:01 2018

@author: J
"""

import numpy as np
import itertools
from scipy import linalg
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
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        print(covar)
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
n_classes = len(np.unique(test_y_red))
gmm         = GaussianMixture(n_components=11, covariance_type='diag')
# gmm.means_  = np.array([test_x_red[test_y_red == i].mean(axis=0)
#                               for i in xrange(n_classes)])
gmm.fit(test_x_red)
# plot_results(test_x_red, gmm.predict(test_x_red), gmm.means_, gmm.covariances_, 0,
#              'Gaussian Mixture')
label   = gmm.predict(test_x_red)
train_accuracy      = np.mean(label == test_y_red) * 100
print(train_accuracy)
# km[i].means_ : # clusters x # features
# km[i].covariances_ : # clusters x # features x # features
"""red"""
for cv_type in ['full', 'tied', 'diag', 'spherical']:
    # gmm         = [GaussianMixture(n_components=i, covariance_type=cv_type) for i in Ks]
    # for j in Ks:
    #     gmm[j].means_  = np.array([test_x_red[test_y_red == i].mean(axis=0)
    #                               for i in xrange(n_classes)])
    # gmm.fit(normalized_x_red)
    # labels = [gmm[i].predict(normalized_x_red) for i in range(len(gmm))]
    # scores = [accuracy_score(test_y_red, labels[i]) for i in range(len(gmm))]
    # print(scores)
    km = [GaussianMixture(n_components=i, covariance_type=cv_type).fit(normalized_x_red) for i in Ks]
    labels = [km[i].predict(test_x_red) for i in range(len(km))]
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
    labels = [km[i].predict(test_x_white) for i in range(len(km))]
    scores = [accuracy_score(test_y_white, labels[i]) for i in range(len(km))]
    print(scores)
    plt.plot(Ks,scores)
    plt.ylabel('accuracy')
    plt.xlabel('# clusters')
    plt.ylim(0,1)
    plt.title('WHITE using covariance type: '+cv_type)
    plt.show()
