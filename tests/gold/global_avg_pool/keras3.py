"""Test suite for the Keras3 module."""

import typing
import keras


class GAPModel(keras.Model):  # type: ignore
  """Test suite for the G A P Model component."""

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return keras.ops.mean(x, axis=(1, 2))  # type: ignore
