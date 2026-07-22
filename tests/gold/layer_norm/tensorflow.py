"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LayerNormModel(tf.keras.Model):
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance."""
    super().__init__()
    self.ln = tf.keras.layers.LayerNormalization(axis=-1)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.ln(x)
