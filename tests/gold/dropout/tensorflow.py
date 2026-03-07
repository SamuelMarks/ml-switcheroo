"""Module docstring."""

import tensorflow as tf


class DropoutModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, p: float = 0.5):
    """Function docstring."""
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(p)

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    return self.dropout(x, training=training)
    # </SWITCHEROO_FAILED_TO_TRANS>
