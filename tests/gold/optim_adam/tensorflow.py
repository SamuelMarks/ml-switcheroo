"""Test suite for the Tensorflow module."""

import tensorflow as tf


def setup_adam(model: tf.keras.Model, lr: float = 0.001):
  """Helper to setup adam."""
  return tf.keras.optimizers.Adam(learning_rate=lr)
