"""Module docstring."""

import keras


def split_tensor(x, split_size: int, axis: int = -1):
  """Function docstring."""
  num_splits = keras.ops.shape(x)[axis] // split_size
  return keras.ops.split(x, num_splits, axis=axis)
