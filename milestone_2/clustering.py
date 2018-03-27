#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:58:01 2018

@author: dohoonkim
"""

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import cdist,pdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

DATASET_PATH_RED = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
DATASET_PATH_WHITE = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-white.csv"

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]

data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_RED,names=data_features)
test_x = data_red[data_features[0:11]]
test_y = data_red["eval"]
normalized_x = StandardScaler().fit_transform(test_x)
"""determine k in k means clustering"""
Ks = range(1, 11)
km = [KMeans(n_clusters=i, random_state=1).fit(normalized_x) for i in Ks]
inertia = [km[i].inertia_ for i in range(len(km))]
clusters_df = pd.DataFrame( { "num_clusters":Ks, "cluster_errors": inertia } )

"""general k-means clustering"""
#print(clusters_df)
#print(inertia)
#plt.plot(Ks,inertia)
#plt.xlabel('k')
#plt.ylabel('explained variance')

#kmeans = KMeans(n_clusters=5, random_state= 40, n_init=10).fit(normalized_x)
#centroids = np.array(kmeans.cluster_centers_)

#plt.scatter(normalized_x[:,8], normalized_x[:,10], c='#050505', s=7)
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(normalized_x[:,0], normalized_x[:,1], normalized_x[:,2])
#ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='#050505', s=5000)
#plt.scatter(centroids[:,8], centroids[:,10], marker='*', s=200, c='g')
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)

"""PCA method for tuning test data"""
mean_vec = np.mean(normalized_x, axis=0)
cov_mat = np.cov(normalized_x.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)
# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 5))
plt.bar(range(11), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(11), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
pca = PCA(n_components=8)
x_8d = pca.fit_transform(normalized_x) #8 dimensional feature space
kmeans = KMeans(n_clusters=5, random_state= 40, n_init=10).fit(x_8d)
centroids = np.array(kmeans.cluster_centers_)
plt.scatter(x_8d[:,0], x_8d[:,1], c='#050505', s=7)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='g')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_8d[:,0], x_8d[:,1], x_8d[:,2])
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='#050505', s=5000)

clusters = kmeans.fit_predict(x_8d)

labels = np.zeros_like(clusters)

for i in range(5):
    mask = (clusters == i)
    labels[mask] = mode(test_y[mask])[0]

print(accuracy_score(test_y, labels))
