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

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)
# data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
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
])  # pyformat: disable


types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())
def dataset(y_name="eval", train_fraction=0.7):
  """Load the imports85 data as a (train,test) pair of `Dataset`.
  Each dataset generates (features_dict, label) pairs.
  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Download and cache the data
  path = sys.argv[1]

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(y_name)

    return features_dict, label

  def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` because `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)

  base_dataset = (
      tf.data
      # Get the lines from the file.
      .TextLineDataset(path)
      # drop lines with question marks.
      .filter(has_no_question_marks))

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line)
           # Cache data so you only decode the file once.
           .cache())

  # Do the same for the test-set.
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test

def to_thousands(features, labels):
    return features, labels / 1

validate_cmdline_args(3,'Usage: python neuralNetwork.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>')
(train, test) = dataset()
train = train.map(to_thousands)
test = test.map(to_thousands)

def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next()
        )
def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())
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
model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
model.train(input_fn=input_train, steps=1000)
eval_result = model.evaluate(input_fn=input_test)
print(eval_result)
average_loss = eval_result["average_loss"]
print(average_loss)
# Convert MSE to Root Mean Square Error (RMSE).
print("\n" + 80 * "*")
print("\nRMS error for the test set: {:.0f}"
    .format(1 * average_loss**0.5))
