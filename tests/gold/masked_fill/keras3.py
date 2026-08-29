"""Test suite for the Keras3 module."""

import typing

import keras


def causal_mask_fill(scores: typing.Any, mask: typing.Any, value: float = -1000000000.0) -> typing.Any:
  """Helper to causal mask fill."""
  return keras.ops.where(mask == 0, value, scores)  # type: ignore
