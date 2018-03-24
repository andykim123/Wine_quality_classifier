import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python decisionTree.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
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

train_y[2<train_y] = 0
train_y[5<=train_y] = 1
train_y[7<=train_y] = 2
test_y[2<test_y] = 0
test_y[5<=test_y] = 1
test_y[7<=test_y] = 2
cv_x = data_red[data_features[:10]]
cv_y = data_red[data_features[11]]
cv_y[2<cv_y] = 0
cv_y[5<=cv_y] = 1
cv_y[7<=cv_y] = 2

print('Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_red_fit.predict(train_x))))
print('Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_red_fit.predict(test_x))))
print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_x, cv_y, cv=10)))

                                    ## White Wine ##
print('-------------White Wine Evaluation-------------')

train_x_w, test_x_w, train_y_w, test_y_w = train_test_split(data_white[data_features[:10]],data_white[data_features[11]], train_size=0.7)

clf_white_fit = clf.fit(train_x_w,train_y_w)

train_w_y[2<train_w_y] = 0
train_w_y[5<=train_w_y] = 1
train_w_y[7<=train_w_y] = 2
test_w_y[2<test_w_y] = 0
test_w_y[5<=test_w_y] = 1
test_w_y[7<=test_w_y] = 2
cv_w_x = data_white[data_features[:10]]
cv_w_y = data_white[data_features[11]]
cv_w_y[2<cv_w_y] = 0
cv_w_y[5<=cv_w_y] = 1
cv_w_y[7<=cv_w_y] = 2

print('Decision Tree Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y_w, clf_white_fit.predict(train_x_w))))
print('Decision Tree Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y_w, clf_white_fit.predict(test_x_w))))
print('Multinomial CV-prediction error rate :: {}'.format(cross_val_score(clf, cv_w_x, cv_w_y, cv=10)))
