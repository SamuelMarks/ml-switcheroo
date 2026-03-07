"""Module docstring."""

import tensorflow as tf


def create_sequential(in_features: int, hidden: int, out_features: int) -> tf.keras.Sequential:
  """Function docstring."""
  return tf.keras.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=(in_features,)),
      tf.keras.layers.Dense(hidden, activation="relu"),
      tf.keras.layers.Dense(out_features),
    ]
  )
