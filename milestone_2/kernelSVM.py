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
validate_cmdline_args(3,'Usage: python kernelSVM.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
DATASET_PATH_RED = sys.argv[1]
DATASET_PATH_WHITE = sys.argv[2]

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_RED,names=data_features)
clf = svm.SVC(decision_function_shape='ovo',kernel='rbf',gamma=2)

                                    ## Red Wine ##
print('-------------Red Wine Evaluation-------------')

train_x, test_x, train_y, test_y = train_test_split(data_red[data_features[:10]],data_red[data_features[11]], train_size=0.7)

clf_red_fit = clf.fit(train_x,train_y)

print('SVM Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_red_fit.predict(train_x))))
print('SVM Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_red_fit.predict(test_x))))
print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data_red[data_features[:10]], data_red[data_features[11]], cv=10)))

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

cv_x = data_red[data_features[:10]]
cv_y = data_red[data_features[11]]
mask = (2 < cv_y) & (cv_y < 5)
cv_y[mask] = 0
mask = (4 < cv_y) & (cv_y < 7)
cv_y[mask] = 1
mask = (6 < cv_y) & (cv_y < 9)
cv_y[mask] = 2

clf_red_fit = clf.fit(train_x,train_y)

print('SVM 3-class Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_red_fit.predict(train_x))))
print('SVM 3-class Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_red_fit.predict(test_x))))
print('3-class CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x, cv_y, cv=10)))


                                    ## White Wine ##
print('-------------White Wine Evaluation-------------')

train_x_w, test_x_w, train_y_w, test_y_w = train_test_split(data_white[data_features[:10]],data_white[data_features[11]], train_size=0.7)

clf_red_fit = clf.fit(train_x_w,train_y_w)

print('SVM Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y_w, clf_red_fit.predict(train_x_w))))
print('SVM Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y_w, clf_red_fit.predict(test_x_w))))
print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, data_red[data_features[:10]], data_red[data_features[11]], cv=10)))

mask = (train_y_w < 5)
train_y_w[mask] = 0
mask = (4 < train_y_w) & (train_y_w < 7)
train_y_w[mask] = 1
mask = (6 < train_y_w)
train_y_w[mask] = 2

mask = (test_y_w < 5)
test_y_w[mask] = 0
mask = (4 < test_y_w) & (test_y_w < 7)
test_y_w[mask] = 1
mask = (6 < test_y_w)
test_y_w[mask] = 2

cv_x_w = data_white[data_features[:10]]
cv_y_w = data_white[data_features[11]]
mask = (cv_y_w < 5)
cv_y_w[mask] = 0
mask = (4 < cv_y_w) & (cv_y_w < 7)
cv_y_w[mask] = 1
mask = (6 < cv_y_w)
cv_y_w[mask] = 2


clf_white_fit = clf.fit(train_x_w,train_y_w)

print('Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y_w, clf_white_fit.predict(train_x_w))))
print('Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y_w, clf_white_fit.predict(test_x_w))))
print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x_w, cv_y_w, cv=10)))
