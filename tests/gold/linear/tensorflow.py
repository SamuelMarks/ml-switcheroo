"""Module docstring."""

import tensorflow as tf


class Model(tf.keras.Model):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.linear = tf.keras.layers.Dense(out_features, input_shape=(in_features,))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.linear(x)
