"""Module docstring."""

import tensorflow as tf


class MaxPoolModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, pool_size: int = 2, strides: int = 2):
    """Function docstring."""
    super().__init__()
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.pool(x)
