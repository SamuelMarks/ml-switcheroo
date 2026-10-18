"""Test suite for the Keras3 module."""

import typing

import keras


def bmm_einsum(x: typing.Any, y: typing.Any) -> typing.Any:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.ops.einsum("bik,bkj->bij", x, y)
