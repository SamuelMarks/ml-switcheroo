"""Module docstring."""

import keras


def bmm_einsum(x, y):
  """Function docstring."""
  # Keras ops
  return keras.ops.einsum("bik,bkj->bij", x, y)
