"""Test suite for the Keras3 module."""

import keras


def create_sequential(in_features: int, hidden: int, out_features: int) -> keras.Sequential:
  """Creates sequential."""
  return keras.Sequential(
    [
      keras.layers.Input(shape=(in_features,)),
      keras.layers.Dense(hidden),
      keras.layers.Activation("relu"),
      keras.layers.Dense(out_features),
    ]
  )
