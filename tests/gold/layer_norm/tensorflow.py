"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: The normalized shape (int) for   init  .
    """
    super().__init__()
    self.ln = tf.keras.layers.LayerNormalization(axis=-1)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.ln(x)
