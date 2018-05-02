import pandas as pd
import numpy as np
import sys
import time
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

def validate_cmdline_args(nargs, msg):
	if len(sys.argv) < nargs:
		print(msg)
		sys.exit(1)
validate_cmdline_args(4,'Usage: python kernelSVM.py <RUN INFILE BOOLEAN> <DATASET_PATH>')

clf = svm.SVC(decision_function_shape='ovo',kernel='rbf', gamma=0.1) #~0.42-0.46

run_infile = False

if(sys.argv[1]=="true" or sys.argv[1]=="True"):
	run_infile = True

data_features = ["f1","f2","f3","f4","f5","f6","f7","f8","score"]


if not run_infile:
	train_x, test_x, train_y, test_y = train_test_split(data[data_features[0:11]],data["eval"], train_size=0.7)
	x_fit = preprocessing.StandardScaler().fit(train_x)
	train_x = x_fit.transform(train_x)
	test_x = x_fit.transform(test_x)
	trainStartTime = time.time()
	clf_fit = clf.fit(train_x,train_y)
	trainTime = time.time() - trainStartTime
	trainTestStartTime = time.time()
	print('SVM Multinomial Classification Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, clf_fit.predict(train_x))))
	trainTestTime = time.time() - trainTestStartTime
	testTestStartTime = time.time()
	print('SVM Multinomial Classification Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, clf_fit.predict(test_x))))
	testTestTime = time.time() - testTestStartTime
	data_x = data[data_features[0:11]]
	x_fit_cv = preprocessing.StandardScaler().fit(data_x)
	data_x_normal = x_fit_cv.transform(data_x)
	cv_result = cross_val_score(clf, data_x_normal, data["eval"], cv=10)
	print('SVM Multinomial CV-prediction error rate :: {}'.format(cv_result))
	print('SVM Multinomial CV-prediction error mean :: {}'.format(np.mean(cv_result)))
	print('SVM Multinomial CV-prediction error variance :: {}'.format(np.var(cv_result)))
	print(trainTime)
	print(trainTestTime)
	print(testTestTime)
else:
	DATASET_PATH_TRAIN = sys.argv[2]
	DATASET_PATH_TEST = sys.argv[3]
	train = pd.read_csv(DATASET_PATH_TRAIN)
	test = pd.read_csv(DATASET_PATH_TEST)
	train_x = train[data_features[0:7]]
	train_y = train["score"]
	test_x = test[data_features[0:7]]
	test_y = test["score"]
	clf_mult_fit = clf.fit(train_x,train_y)
	print(metrics.accuracy_score(test_y, clf_mult_fit.predict(test_x)))

	# x_fit = preprocessing.StandardScaler().fit(data[data_features[0:11]])
	# data_x = x_fit.transform(data[data_features[0:11]])
	# print(cross_val_score(clf, data_x, data["eval"], cv=10))
