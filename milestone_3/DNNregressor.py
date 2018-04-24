from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
import sys
import collections
try:
    import pandas as pd
except ImportError:
    pass

# disable warning messages:
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# functions:
def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)

def dataset(y_name="eval", train_fraction=0.7, dataset_path=""):
  path = dataset_path

  def decode_line(line):
    items = tf.decode_csv(line, list(defaults.values()))
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)
    label = features_dict.pop(y_name)
    return features_dict, label

  def in_training_set(line):
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    return ~in_training_set(line)

  base_dataset = (tf.data.TextLineDataset(path))
  train = (base_dataset.filter(in_training_set).map(decode_line).cache())
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test

def input_train():
    return (train.shuffle(1000).batch(128).repeat()
		.make_one_shot_iterator().get_next())

def input_test():
    return (test.shuffle(1000).batch(128)
		.make_one_shot_iterator().get_next())

def to_thousands(features, labels):
    return features, labels / 1000


# defining dataset object:
defaults = collections.OrderedDict([
    ("fa", [0.0]),
    ("va", [0.0]),
    ("ca", [0.0]),
    ("rs", [0.0]),
    ("ch", [0.0]),
    ("fsd", [0.0]),
    ("tsd", [0.0]),
    ("dens", [0.0]),
    ("pH", [0.0]),
    ("sulp", [0.0]),
    ("alcohol", [0.0]),
    ("eval", [0])
])

feature_columns = [tf.feature_column.numeric_column(key="fa"),
                   tf.feature_column.numeric_column(key="va"),
                   tf.feature_column.numeric_column(key="ca"),
                   tf.feature_column.numeric_column(key="rs"),
                   tf.feature_column.numeric_column(key="ch"),
                   tf.feature_column.numeric_column(key="fsd"),
                   tf.feature_column.numeric_column(key="tsd"),
                   tf.feature_column.numeric_column(key="dens"),
                   tf.feature_column.numeric_column(key="pH"),
                   tf.feature_column.numeric_column(key="sulp"),
                   tf.feature_column.numeric_column(key="alcohol"),
                   ]

types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())

# our code runs here:
validate_cmdline_args(3,'Usage: python neuralNetwork.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
hidden_layers = []

specify = raw_input("Would you like to specify network layers (y/n): ")
if str(specify) == "y":
	var = input("Number of hidden layers: ")
	for i in range (0, int(var)):
		nodes = input("Layer " + str(i+1) + " num nodes: ")
		hidden_layers.append(nodes)
		print("running DNN with hidden units " + str(hidden_layers) + " ...")
else:
	hidden_layers = [4, 4, 4, 4, 4]
	print("running DNN with default hidden units " + str(hidden_layers) + " ...")


						## Red Wine ##
(train, test) = dataset(dataset_path=sys.argv[1])
train = train.map(to_thousands)
test = test.map(to_thousands)

model = tf.estimator.DNNRegressor(hidden_units=hidden_layers, feature_columns=feature_columns)
model.train(input_fn=input_train, steps=1000)
eval_result = model.evaluate(input_fn=input_test)

print("\n" + 30 * "*" + "DNN RESULTS" + 30 * "*")
print("Red wine loss: "+str(eval_result["loss"]))
print("Red wine average loss: "+str(eval_result["average_loss"]))
# Convert MSE to Root Mean Square Error (RMSE).
print("RMS error for the test set: {:.0f}".format(1000 * eval_result["average_loss"]**0.5))

						## White Wine ##
(train, test) = dataset(dataset_path=sys.argv[2])
train = train.map(to_thousands)
test = test.map(to_thousands)

model = tf.estimator.DNNRegressor(hidden_units=hidden_layers, feature_columns=feature_columns)
model.train(input_fn=input_train, steps=1000)
eval_result = model.evaluate(input_fn=input_test)
print("\nWhite wine loss: "+str(eval_result["loss"]))
print("White wine average loss: "+str(eval_result["average_loss"]))
print("RMS error for the test set: {:.0f}".format(1000 * eval_result["average_loss"]**0.5))
print(80 * "*" + "\n")
