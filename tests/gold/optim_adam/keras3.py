"""Test suite for the Keras3 module."""

import typing

import keras


def setup_adam(model: keras.Model, lr: float = 0.001) -> typing.Any:  # type: ignore
  """Helper to setup adam."""
  return keras.optimizers.Adam(learning_rate=lr)  # type: ignore
