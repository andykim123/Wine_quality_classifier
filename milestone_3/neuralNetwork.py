from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


try:
    import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
    pass


def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
validate_cmdline_args(3,'Usage: python neuralNetwork.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
DATASET_PATH_RED = sys.argv[1]
DATASET_PATH_WHITE = sys.argv[2]

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data_red = pd.read_csv(DATASET_PATH_RED,names=data_features)
data_white = pd.read_csv(DATASET_PATH_RED,names=data_features)
model = tf.estimator.LinearRegressor(feature_columns=data_features)
# train_x, test_x, train_y, test_y = train_test_split(data_red[data_features[:10]],data_red[data_features[11]], train_size=0.7)
x_train = data_red.sample(frac=0.7, random_state=None)
y_train = x_train.pop("eval")
x_test = data_red.drop(x_train.index)
y_test = x_test.pop("eval")
train_x = list((x_train, y_train))
test_x = list((x_test, y_test))
print(train_x)
print(test_x)
# def to_thousands(features, labels):
#     return features, labels / 1000
#
# train = x_train.map(to_thousands)
# test = x_test.map(to_thousands)

def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train_x.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())
                ## Red Wine ##
print('-------------Red Wine Evaluation-------------')
model.train(input_fn=input_train, steps=1000)
model.train(input_fn=input_train, steps=1000)
# model.train(x_train, 1000)
eval_result = model.evaluate(test)
