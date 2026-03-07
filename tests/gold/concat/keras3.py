"""Module docstring."""

import keras


def concat_tensors(x, y, axis: int = -1):
  """Function docstring."""
  # Keras 3 (backend agnostic) uses keras.ops
  return keras.ops.concatenate([x, y], axis=axis)
