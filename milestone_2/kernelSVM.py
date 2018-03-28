import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python kernelSVM.py <DATASET_PATH> <RUN INFILE BOOLEAN>')
# run_infile_boolean is a boolean which checks whether a particular run is done within other python file or not
# if it is true, it indicates that the run in done withtin other python file run. If false, it is done in command line.
DATASET_PATH = sys.argv[1]
run_infile = sys.argv[2]

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)
#clf = svm.SVC(decision_function_shape='ovo',kernel='linear') #~0.5-0.63
clf = svm.SVC(decision_function_shape='ovo',kernel='poly',degree=2,coef0=10) #~0.48-0.62
#clf = svm.SVC(decision_function_shape='ovo',kernel='rbf', gamma=4) #~0.42-0.46
#clf = svm.SVC(decision_function_shape='ovo',kernel='sigmoid', coef0=20) #~0.42

run_infile = False

if(sys.argv[2]=="true" or sys.argv[2]=="True"):
    run_infile = True

if not run_infile:
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
    clf_fit = clf.fit(train_x,train_y)
    print('SVM Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_fit.predict(train_x))))
    print('SVM Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_fit.predict(test_x))))
    print('SVM Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10)))
    # data set modification to 3-class classification. one for 3/4, one for 5/6, one for 7/8
    mask = (2 < train_y) & (train_y < 5)
    train_y[mask] = 0
    mask = (4 < train_y) & (train_y < 7)
    train_y[mask] = 1
    mask = (6 < train_y) & (train_y < 9)
    train_y[mask] = 2
    mask = (2 < test_y) & (test_y < 5)
    test_y[mask] = 0
    mask = (4 < test_y) & (test_y < 7)
    test_y[mask] = 1
    mask = (6 < test_y) & (test_y < 9)
    test_y[mask] = 2
    cv_x = data[data_features[:10]]
    cv_y = data[data_features[11]]
    mask = (2 < cv_y) & (cv_y < 5)
    cv_y[mask] = 0
    mask = (4 < cv_y) & (cv_y < 7)
    cv_y[mask] = 1
    mask = (6 < cv_y) & (cv_y < 9)
    cv_y[mask] = 2
    clf_fit = clf.fit(train_x,train_y)
    print('SVM 3-class Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_fit.predict(train_x))))
    print('SVM 3-class Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_fit.predict(test_x))))
    print('3-class CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x, cv_y, cv=10)))
else:
	#if the run in done within modelEvaluation.py, we just return cross_val_score result, which is a list of 10 different float-type accuracies
    train_x, test_x, train_y, test_y = train_test_split(data[data_features[:10]],data[data_features[11]], train_size=0.7)
    clf_fit = clf.fit(train_x,train_y)
    print(cross_val_score(clf, data[data_features[:10]], data[data_features[11]], cv=10))
