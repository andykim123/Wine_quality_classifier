from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import sys
import numpy as np
import tensorflow as tf

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

                            ## Red Wine ##
print('-------------Red Wine Evaluation-------------')

np.random.seed(None)

x_train = data_red.sample(frac=0.7, random_state=None)
x_test = data_red.drop(x_train.index)
y_train = x_train.pop("eval")
y_test = x_test.pop("eval")

train = (x_train, y_train)
test = (x_test, y_test)

train.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next()

print(train)
print(test)

