import pandas as pd
import numpy as np
import sys
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python adaBoost.py <RUN INFILE BOOLEAN> <DATASET_PATH> n_estimator')

run_infile = False

if(sys.argv[1]=="true" or sys.argv[1]=="True"):
    run_infile = True

#if(n>10):
#	print("The number of estimators should be equal or less than 10, which is the number of total features.")
#	sys.exit(1)

data_features = ["f1","f2","f3","f4","f5","f6","f7","f8","score"]

if not run_infile:
    if len(sys.argv)==3:
        n = 8
    elif len(sys.argv)==4:
        n = int(sys.argv[3])
        
    data = pd.read_csv(DATASET_PATH,names=data_features)
    clf = RandomForestClassifier(n_estimators=n)
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:7]],data[data_features[8]], train_size=0.7)
    clf_mult_fit = clf.fit(train_x,train_y)
    print('Random Forest Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_x))))
    print('Random Forest Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_x))))
    print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10)))
    train_y[train_y<5] = 0
    train_y[train_y>=5] = 1
    test_y[test_y<5] = 0
    test_y[test_y>=5] = 1
    cv_x = data_red[data_features[:10]]
    cv_y = data_red[data_features[11]]
    cv_y[cv_y<5] = 0
    cv_y[cv_y>=5] = 1
    clf_bin_fit = clf.fit(train_x, train_y)
    print('Random Forest Decision Tree Binary Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_bin_fit.predict(train_x))))
    print('Random Forest Decision Tree Binary Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_bin_fit.predict(test_x))))
    print('Binary CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x, cv_y, cv=10)))
else:
    n = 8
    DATASET_PATH_TRAIN = sys.argv[2]
    DATASET_PATH_TEST = sys.argv[3]
    train = pd.read_csv(DATASET_PATH_TRAIN)
    test = pd.read_csv(DATASET_PATH_TEST)
    train_x = train[data_features[0:7]]
    train_y = train["score"]
    test_x = test[data_features[0:7]]
    test_y = test["score"]
    clf = RandomForestClassifier(n_estimators=n)
    clf_mult_fit = clf.fit(train_x,train_y)
    print(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_x)))