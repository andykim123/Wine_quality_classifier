import pandas as pd
import numpy as np
import sys
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(4,'Usage: python randomForest.py <DATASET_PATH> n_estimator <RUN INFILE BOOLEAN>')
DATASET_PATH = sys.argv[1]
n = int(sys.argv[2])

if(n>10):
	print("The number of estimators should be equal or less than 10, which is the number of total features.")
	sys.exit(1)

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)
clf = RandomForestClassifier(n_estimators=n)

run_infile = False

if(sys.argv[3]=="true" or sys.argv[3]=="True"):
    run_infile = True


if not run_infile:
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
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
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
    clf_fit = clf.fit(train_x,train_y)
    print(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10))