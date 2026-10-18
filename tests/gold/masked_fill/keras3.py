"""Test suite for the Keras3 module."""

import typing

import keras


def causal_mask_fill(scores: typing.Any, mask: typing.Any, value: float = -1000000000.0) -> typing.Any:
  """Helper to causal mask fill.

  Args:
      mask: Test argument or fixture.
      scores: Test argument or fixture.
      value: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.ops.where(mask == 0, value, scores)
