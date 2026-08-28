"""Test suite for the Keras3 module."""

import typing
import keras


def relu_activation(x: typing.Any) -> typing.Any:
  """Helper to relu activation."""
  return keras.activations.relu(x)  # type: ignore
