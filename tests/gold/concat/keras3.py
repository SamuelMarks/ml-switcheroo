"""Test suite for the Keras3 module."""

import keras


def concat_tensors(x, y, axis: int = -1):
  """Helper to concat tensors."""
  return keras.ops.concatenate([x, y], axis=axis)
