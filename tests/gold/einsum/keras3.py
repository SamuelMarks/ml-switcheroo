"""Test suite for the Keras3 module."""

import typing
import keras


def bmm_einsum(x: typing.Any, y: typing.Any) -> typing.Any:
  """Helper to bmm einsum."""
  return keras.ops.einsum("bik,bkj->bij", x, y)  # type: ignore
