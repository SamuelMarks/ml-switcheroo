"""Test suite for the Keras3 module."""

import typing

import keras


def gelu_activation(x: typing.Any, approximate: bool = False) -> typing.Any:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.activations.gelu(x, approximate=approximate)
