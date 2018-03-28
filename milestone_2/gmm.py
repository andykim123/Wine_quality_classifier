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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

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
test_x_red_23 = data_red[data_features[1:3]]
print(test_x_red_35)
test_y_red = data_red["eval"]
test_x_white = data_white[data_features[0:11]]
test_y_white = data_white["eval"]
normalized_x_red = StandardScaler().fit_transform(test_x_red)
normalized_x_red_23 = StandardScaler().fit_transform(test_x_red_23)
normalized_x_white = StandardScaler().fit_transform(test_x_white)
"""
'full' (each component has its own general covariance matrix),
'tied' (all components share the same general covariance matrix),
'diag' (each component has its own diagonal covariance matrix),
'spherical' (each component has its own single variance).
"""
Ks = range(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(normalized_x_red)
          for n in Ks]

plt.plot(Ks, [m.bic(normalized_x_red) for m in models], label='BIC')
plt.plot(Ks, [m.aic(normalized_x_red) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('Ks');
plt.show()
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        # print(covariance[:,0:1])
        # width = 2 * np.sqrt(abs(covariance[:,0]))
        # height = 2 * np.sqrt(abs(covariance[:,1]))
        width, height = 2 * np.sqrt(covariance)
        print(width)
        print(height)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,angle, **kwargs))
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    print(X[:,1])
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
gmx = GaussianMixture(n_components=8, covariance_type='full', random_state=42)
gmx_23 = GaussianMixture(n_components=8, covariance_type='full', random_state=42)
# gmx = GaussianMixture(n_components=8, covariance_type='diag', random_state=42)
# gmx = GaussianMixture(n_components=8, covariance_type='tied')
# gmx = GaussianMixture(n_components=8, covariance_type='spherical')
plot_gmm(gmx_23, normalized_x_red_23)
labels = gmx.fit(normalized_x_red).predict(normalized_x_red)
centroids = gmx.means_
# plt.scatter(normalized_x_red[:, 0], normalized_x_red[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
# plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, label='centroids', c='g')
# plt.legend()
# plt.ylabel('feature 0')
# plt.xlabel('feature 1')
# # plt.title('WHITE using covariance type: '+cv_type)
# plt.show()

plt.scatter(normalized_x_red[:,2], normalized_x_red[:,3], c='#050505', s=7)
plt.scatter(centroids[:,2], centroids[:,3], marker='*', label='centroids', s=200, c='g')
plt.legend()
plt.ylabel('feature 2')
plt.xlabel('feature 3')
plt.show()

plt.scatter(normalized_x_red[:,2], normalized_x_red[:,3], c=labels, s=7)
plt.scatter(centroids[:,2], centroids[:,3], marker='*', label='centroids', s=200, c='g')
plt.legend()
plt.ylabel('feature 2')
plt.xlabel('feature 3')
plt.show()
# plt.scatter(normalized_x_red[:,3], normalized_x_red[:,5], c=labels, s=7)
# plt.scatter(centroids[:,3], centroids[:,5], marker='*', label='centroids', s=200, c='g')
# plt.legend()
# plt.ylabel('feature 3')
# plt.xlabel('feature 5')
# # plt.title('WHITE using covariance type: '+cv_type)
# plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(normalized_x_red[:,2], normalized_x_red[:,3], normalized_x_red[:,5])
ax.scatter(centroids[:, 2], centroids[:, 3], centroids[:, 5], marker='*', label='centroids', c='#050505', s=5000)
# ax.show()
# print(gmx.means_)
plt.show()
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
    # print(scores)
    # plt.plot(Ks,scores)
    # plt.ylabel('accuracy')
    # plt.xlabel('# clusters')
    # plt.ylim(0,1)
    # plt.title('RED using covariance type: '+cv_type)
    # plt.show()
# ===================================================================================================
"""white"""
for cv_type in ['full', 'tied', 'diag', 'spherical']:
    km = [GaussianMixture(n_components=i, covariance_type=cv_type).fit(normalized_x_white) for i in Ks]
    labels = [km[i].predict(test_x_white) for i in range(len(km))]
    scores = [accuracy_score(test_y_white, labels[i]) for i in range(len(km))]
    # print(scores)
    # plt.plot(Ks,scores)
    # plt.ylabel('accuracy')
    # plt.xlabel('# clusters')
    # plt.ylim(0,1)
    # plt.title('WHITE using covariance type: '+cv_type)
    # plt.show()
