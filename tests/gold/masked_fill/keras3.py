"""Test suite for the Keras3 module."""

import keras


def causal_mask_fill(
  scores: keras.KerasTensor, mask: keras.KerasTensor, value: float = -1000000000.0
) -> keras.KerasTensor:
  """Execute `where` to support causal mask fill in keras3 tests.

  Args:
      mask: Input `mask` parameter value (`keras.KerasTensor`).
      scores: Input `scores` parameter value (`keras.KerasTensor`).
      value: Input `value` parameter value (`float`).

  Returns:
      Resulting `keras.KerasTensor` from causal mask fill.
  """
  return keras.ops.where(mask == 0, value, scores)
