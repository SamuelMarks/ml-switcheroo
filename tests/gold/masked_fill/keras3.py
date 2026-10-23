"""Test suite for the Keras3 module."""

import keras


def causal_mask_fill(
  scores: keras.KerasTensor, mask: keras.KerasTensor, value: float = -1000000000.0
) -> keras.KerasTensor:
  """Helper to causal mask fill.

  Args:
      mask: The mask (keras.KerasTensor) for causal mask fill.
      scores: The scores (keras.KerasTensor) for causal mask fill.
      value: The value (float) for causal mask fill.

  Returns:
      Resulting `keras.KerasTensor` from causal mask fill.
  """
  return keras.ops.where(mask == 0, value, scores)
