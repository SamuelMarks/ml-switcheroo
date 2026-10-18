"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
    """
    super().__init__()
    self.ln = tf.keras.layers.LayerNormalization(axis=-1)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
