"""Test suite for the Tensorflow module."""

import tensorflow as tf


def create_sequential(in_features: int, hidden: int, out_features: int) -> tf.keras.Sequential:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.keras.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=(in_features,)),
      tf.keras.layers.Dense(hidden, activation="relu"),
      tf.keras.layers.Dense(out_features),
    ]
  )
