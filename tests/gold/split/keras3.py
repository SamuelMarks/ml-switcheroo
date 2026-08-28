"""Test suite for the Keras3 module."""

import typing
import keras


def split_tensor(x: typing.Any, split_size: int, axis: int = -1) -> typing.Any:
  """Splits tensor."""
  num_splits = keras.ops.shape(x)[axis] // split_size  # type: ignore
  return keras.ops.split(x, num_splits, axis=axis)  # type: ignore
