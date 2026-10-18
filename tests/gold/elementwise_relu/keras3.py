"""Test suite for the Keras3 module."""

import typing

import keras


def relu_activation(x: typing.Any) -> typing.Any:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.activations.relu(x)
