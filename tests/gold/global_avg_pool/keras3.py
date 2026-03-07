"""Module docstring."""

import keras


class GAPModel(keras.Model):
  """Class docstring."""

  def call(self, x):
    """Function docstring."""
    return keras.ops.mean(x, axis=(1, 2))
