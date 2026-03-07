"""Module docstring."""

import tensorflow as tf


class BNModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, num_features: int):
    """Function docstring."""
    super().__init__()
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    """Function docstring."""
    return self.bn(x, training=training)
