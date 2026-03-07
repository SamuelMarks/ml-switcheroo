"""Module docstring."""

import tensorflow as tf


class FlattenModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.flatten = tf.keras.layers.Flatten()
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.flatten(x)
