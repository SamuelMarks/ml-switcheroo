"""Module docstring."""

import tensorflow as tf


class GAPModel(tf.keras.Model):
  """Class docstring."""

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return tf.math.reduce_mean(x, axis=(1, 2))
