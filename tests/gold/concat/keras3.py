"""Test suite for the Keras3 module."""

import typing

import keras


def concat_tensors(x: typing.Any, y: typing.Any, axis: int = -1) -> typing.Any:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.ops.concatenate([x, y], axis=axis)
