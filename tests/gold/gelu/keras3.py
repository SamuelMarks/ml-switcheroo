"""Test suite for the Keras3 module."""

import typing

import keras


def gelu_activation(x: typing.Any, approximate: bool = False) -> typing.Any:
  """Helper to gelu activation."""
  return keras.activations.gelu(x, approximate=approximate)  # type: ignore
