"""Test suite for the Keras3 module."""

import typing

import keras


def concat_tensors(x: typing.Any, y: typing.Any, axis: int = -1) -> typing.Any:
  """Helper to concat tensors."""
  return keras.ops.concatenate([x, y], axis=axis)  # type: ignore
