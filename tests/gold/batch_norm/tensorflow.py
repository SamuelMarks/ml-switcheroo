"""Test suite for the Tensorflow module."""

import tensorflow as tf


class BNModel:
  """Test double `BNModel` implementing `call` for tensorflow tests."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: Input `num_features` parameter value (`int`).
    """
    super().__init__()
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    """Execute `bn` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.bn(x, training=training)
