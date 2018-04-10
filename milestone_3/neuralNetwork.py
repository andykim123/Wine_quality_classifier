from __future__ import absolute_import, division, print_function

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

<<<<<<< HEAD

try:
    import pandas as pd
except ImportError:
    pass


=======
>>>>>>> parent of 765eabb... split test and train
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

<<<<<<< HEAD
np.random.seed(None)

model = tf.estimator.LinearRegressor(feature_columns=data_features)

x_train = data_red.sample(frac=0.7, random_state=None)
y_train = x_train.pop("eval")
x_test = data_red.drop(x_train.index)
y_train = x_train.pop("eval")
y_test = x_test.pop("eval")

train = (x_train, y_train)
test = (x_test, y_test)

train.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next()
=======
>>>>>>> parent of 765eabb... split test and train

print(train)
print(test)
