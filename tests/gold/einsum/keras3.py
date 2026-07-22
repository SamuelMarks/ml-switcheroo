"""Test suite for the Keras3 module."""

import keras


def bmm_einsum(x, y):
  """Helper to bmm einsum."""
  return keras.ops.einsum("bik,bkj->bij", x, y)
