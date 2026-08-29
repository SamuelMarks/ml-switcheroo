"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


def setup_adam(model: tf.keras.Model, lr: float = 0.001) -> typing.Any:  # type: ignore
  """Helper to setup adam."""
  return tf.keras.optimizers.Adam(learning_rate=lr)  # type: ignore
