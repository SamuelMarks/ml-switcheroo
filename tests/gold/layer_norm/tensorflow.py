"""Module docstring."""

import tensorflow as tf


class LayerNormModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, normalized_shape: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.ln = tf.keras.layers.LayerNormalization(axis=-1)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.ln(x)
