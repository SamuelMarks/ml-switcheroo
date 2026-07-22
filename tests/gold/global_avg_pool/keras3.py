"""Test suite for the Keras3 module."""

import keras


class GAPModel(keras.Model):
  """Test suite for the G A P Model component."""

  def call(self, x):
    """Helper to call."""
    return keras.ops.mean(x, axis=(1, 2))
