"""Module docstring."""

import keras


def causal_mask_fill(scores, mask, value: float = -1e9):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return keras.ops.where(mask == 0, value, scores)
  # </SWITCHEROO_FAILED_TO_TRANS>
