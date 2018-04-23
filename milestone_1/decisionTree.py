import pandas as pd
import numpy as np
import sys
import time
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python decisionTree.py <DATASET_PATH> <RUN INFILE BOOLEAN>')
DATASET_PATH = sys.argv[1]

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)
clf = tree.DecisionTreeClassifier()

run_infile = False

if(sys.argv[2]=="true" or sys.argv[2]=="True"):
    run_infile = True

if not run_infile:
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
    trainStartTime = time.time();
    clf_mult_fit = clf.fit(train_x,train_y)
    trainTime = time.time()-trainStartTime;
    print('Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_mult_fit.predict(train_x))))
    testStartTime = time.time();
    print('Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_x))))
    testTime = time.time() - testStartTime;
    print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10)))
    print('Train Time: '+str(trainTime));
    print('Test Time: '+str(testTime));
#    train_y[train_y<5] = 0
#    train_y[train_y>=5] = 1
#    test_y[test_y<5] = 0
#    test_y[test_y>=5] = 1
#    cv_x = data[data_features[:10]]
#    cv_y = data[data_features[11]]
#    cv_y[cv_y<5] = 0
#    cv_y[cv_y>=5] = 1
#    clf_bin_fit = clf.fit(train_x, train_y)
#    print('Decision Tree Binary Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_bin_fit.predict(train_x))))
#    print('Decision Tree Binary Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_bin_fit.predict(test_x))))
#    print('Binary CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x, cv_y, cv=10)))
else:
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
    print(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10))