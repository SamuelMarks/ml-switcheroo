"""Test suite for the Tensorflow module."""

import tensorflow as tf


def create_sequential(in_features: int, hidden: int, out_features: int) -> tf.keras.Sequential:
  """Creates sequential.

  Args:
      in_features: Number of input linear features.
      hidden: The hidden (int) for create sequential.
      out_features: Number of output linear features.

  Returns:
      Resulting `tf.keras.Sequential` from create sequential.
  """
  return tf.keras.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=(in_features,)),
      tf.keras.layers.Dense(hidden, activation="relu"),
      tf.keras.layers.Dense(out_features),
    ]
  )
