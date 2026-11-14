"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LayerNormModel:
  """Test double `LayerNormModel` implementing `call` for tensorflow tests."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Input `normalized_shape` parameter value (`int`).
    """
    super().__init__()
    self.ln = tf.keras.layers.LayerNormalization(axis=-1)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `ln` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.ln(x)
