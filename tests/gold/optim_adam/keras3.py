"""Test suite for the Keras3 module."""

import keras


def setup_adam(model: keras.Model, lr: float = 0.001):
  """Helper to setup adam."""
  return keras.optimizers.Adam(learning_rate=lr)
