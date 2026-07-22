"""Test suite for the Keras3 module."""

import keras


def causal_mask_fill(scores, mask, value: float = -1000000000.0):
  """Helper to causal mask fill."""
  return keras.ops.where(mask == 0, value, scores)
