#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:09:32 2018

@author: dohoonkim
"""
import pandas as pd
import numpy as np
import sys
import csv
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import gaussian_process
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

DATASET_PATH_RED = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-red.csv"
DATASET_PATH_WHITE = "/Users/dohoonkim/Desktop/cse517a/ApplicationProject/winequality-white.csv"

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]

data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_RED,names=data_features)

#train_x, test_x, train_y, test_y = train_test_split(data_red[data_features[0:11]],data_red["eval"], train_size=0.7)
#"""normalize the data for PCA"""
#normalized_test = StandardScaler().fit_transform(test_x)
#normalized_train = StandardScaler().fit_transform(train_x)
#"""proof for 8 components with explained variance, 100% of attained variance for 8 components and 90% of attained variance for 6 components"""
#pca = PCA().fit(normalized_train)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');

"""training set and test set with 8 components"""
#pca = PCA(n_components=8)
#pca = pca.fit(normalized_train)
#train_8d = pca.transform(normalized_train)
#test_8d = pca.transform(normalized_test)
#
#train_data = pd.DataFrame(data={"f1":train_8d[:,0], "f2":train_8d[:,1], "f3":train_8d[:,2],"f4":train_8d[:,3],"f5":train_8d[:,4],"f6":train_8d[:,5],"f7":train_8d[:,6],"f8":train_8d[:,7],"score":train_y})
#test_data = pd.DataFrame(data={"f1":test_8d[:,0], "f2":test_8d[:,1], "f3":test_8d[:,2],"f4":test_8d[:,3],"f5":test_8d[:,4],"f6":test_8d[:,5],"f7":test_8d[:,6],"f8":test_8d[:,7],"score":test_y})


#train_data.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAtrain.csv", sep=',',index=False)
#test_data.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAtest.csv", sep=',',index=False)

#train_pca = pd.read_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAtrain.csv")
#test_pca = pd.read_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAtest.csv")
"""for CV, PCA on whole dataset"""
pca = PCA(n_components=8)
normalized_data = StandardScaler().fit_transform(data_red[data_features[0:11]])
data_8d = pca.fit_transform(normalized_data)
PCA_data = pd.DataFrame(data={"f1":data_8d[:,0], "f2":data_8d[:,1], "f3":data_8d[:,2],"f4":data_8d[:,3],"f5":data_8d[:,4],"f6":data_8d[:,5],"f7":data_8d[:,6],"f8":data_8d[:,7],"score":data_red["eval"]})
#Save pca version data to csv
PCA_data.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAdata.csv", sep=',',index=False)
#csv to dataframe
data_pca = pd.read_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAdata.csv")
"""shuffle the data"""
#data_pca = data_pca.reindex(np.random.permutation(data_pca.index))
#data_pca.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/shuffled.csv",index=False)
#data_pca = pd.read_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/shuffled.csv")
#shuffled = shuffle(data_pca)
#shuffled.sort_index()
"""data split for cross validation""" 
"""create 10 test data sets(10%) and corresponding 10 train datasets(90%)"""
#testdata = data_pca[160:320] #160~319
#excluded = list(range(160,320)) #160~319
#bad_df = data_pca.index.isin(excluded)
#newdata = data_pca[~bad_df]
#traindata = newdata
#testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%d.csv" %1,index=False)
#i = range(10)
"""split 10 sets of training and testing after being shuffled and save to csv files"""
def split(data_pca):
    data_pca = data_pca.reindex(np.random.permutation(data_pca.index))
    data_pca.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/shuffled.csv",index=False)
    data_pca = pd.read_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/shuffled.csv")
    for i in range(0,10):
        """for last test set, we grab 159 data points"""
        if i==9:
            testdata = data_pca[160*i:1599]
            excluded = list(range(160*i,1599))
            bad_df = data_pca.index.isin(excluded)
            newdata = data_pca[~bad_df]
            traindata = newdata
            testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%d.csv"%i,index=False)
            traindata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/train%d.csv"%i,index=False)
        else: #0~8 for other cases, 160 data points
            testdata = data_pca[160*i:(i+1)*160]
            excluded = list(range(160*i,(i+1)*160))
            bad_df = data_pca.index.isin(excluded)
            newdata = data_pca[~bad_df]
            traindata = newdata        
            testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%d.csv"%i,index=False)
            traindata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/train%d.csv"%i,index=False)
#def split(data_pca):
#    for i in range(1,10):
#        if i==9:
#            testdata = data_pca[160*i:1599]
#            excluded = list(range(160*i,1599))
#            bad_df = data_pca.index.isin(excluded)
#            newdata = data_pca[~bad_df]
#            traindata = newdata
#            testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%d.csv"%i,index=False)
#            traindata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/train%d.csv"%i,index=False)
#        else: #0~8
#            testdata = data_pca[160*i:(i+1)*160]
#            excluded = list(range(160*i,(i+1)*160))
#            bad_df = data_pca.index.isin(excluded)
#            newdata = data_pca[~bad_df]
#            traindata = newdata        
#            testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%d.csv"%i,index=False)
#            traindata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/train%d.csv"%i,index=False)
#        
#def split(data_pca):
#    for i in range(10):
#        testdata = data_pca[160*i:i+160,:]
#        excluded = list(range(160*i,i+160))
#        bad_df = data_pca.index.isin(excluded)
#        newdata = data_pca[~bad_df]
#        traindata = newdata
#        testdata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/test%.csv"%i)
#        traindata.to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/train%.csv"%i)


"""visualization of 2D PCA-components with label outputs"""
#colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'seagreen', 'khaki', 'navy']
#
#quality = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#lw = 2
#
#for color, i in zip(colors, quality):
#    plt.scatter(train_8d[train_y == i, 6], train_8d[train_y == i, 7], color=color, alpha=.8, lw=lw,
#                label=i)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('PCA of wine training dataset')
#
#plt.figure()

"""Multinomial with PCA"""
#lr = linear_model.LogisticRegression()
#mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
#lr.fit(train_8d, train_y)
#mul_lr_fit = mul_lr.fit(train_8d, train_y)
#print('Multinomial Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, mul_lr_fit.predict(train_8d))))
#print('Multinomial Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, mul_lr_fit.predict(test_8d))))
#scores = cross_val_score(mul_lr, data_8d, data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""decision tree"""
#clf = tree.DecisionTreeClassifier()
#clf_mult_fit = clf.fit(train_8d,train_y)
#print('Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_8d))))
#print('Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_8d))))
#scores = cross_val_score(clf, data_red[data_features[0:11]], data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""decision tree bagging"""
#clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
#clf_mult_fit = clf.fit(train_8d,train_y)
#print('Decision Tree with Bagging Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_8d))))
#print('Decision Tree with Bagging Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_8d))))
#scores = cross_val_score(clf, data_red[data_features[0:11]], data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""Random Forest 5"""
#clf = RandomForestClassifier(n_estimators=5)
#clf_mult_fit = clf.fit(train_8d,train_y)
#print('Random Forest Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_8d))))
#print('Random Forest Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_8d))))
#print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data_8d, data_red["eval"], cv=10)))
#scores = cross_val_score(clf, data_8d, data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""Adaboost 5"""
#clf = AdaBoostClassifier(n_estimators=5)
#clf_mult_fit = clf.fit(train_8d,train_y)
#print('Adaboost Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_8d))))
#print('Adaboost Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_8d))))
#scores = cross_val_score(clf, data_red[data_features[0:11]], data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""Gaussian Process"""
"""1-vs-1 classification of multivariate GP"""
"""RBF Kernel"""
#mul_gp2 = gaussian_process.GaussianProcessClassifier(multi_class='one_vs_one').fit(train_x,train_y)
#print('Multiclass (1-vs-1) Gaussian Process Train Accuracy :: %.3f\n' % (metrics.accuracy_score(train_y, mul_gp2.predict(train_8d))))
#print('Multiclass (1-vs-1) Gaussian Process Test Accuracy :: %.3f\n' % (metrics.accuracy_score(test_y, mul_gp2.predict(test_8d))))
#cv_gp2 = cross_val_score(mul_gp2, data_8d, data_red["eval"], cv=10)
#print("Accuracy(Mean CV): %0.2f (+/- %0.2f)" % (cv_gp2.mean(), cv_gp2.std() * 2))

"""Kernel SVM"""
#clf = svm.SVC(decision_function_shape='ovo',kernel='rbf', gamma=0.1)
#clf_fit = clf.fit(train_8d,train_y)
#print('SVM Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_fit.predict(train_8d))))
#print('SVM Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_fit.predict(test_8d))))
#scores = cross_val_score(clf, data_8d, data_red["eval"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))