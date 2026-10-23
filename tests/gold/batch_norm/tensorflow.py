"""Test suite for the Tensorflow module."""

import tensorflow as tf


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: The num features (int) for   init  .
    """
    super().__init__()
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool | None) for call.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.bn(x, training=training)
