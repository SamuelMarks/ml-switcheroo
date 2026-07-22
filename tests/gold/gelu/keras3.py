"""Test suite for the Keras3 module."""

import keras


def gelu_activation(x, approximate: bool = False):
  """Helper to gelu activation."""
  return keras.activations.gelu(x, approximate=approximate)
