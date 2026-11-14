"""Test suite for the Tensorflow module."""

import tensorflow as tf


class DropoutModel:
  """Test double `DropoutModel` implementing `call` for tensorflow tests."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: Input `p` parameter value (`float`).
    """
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(p)

  def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    """Execute `dropout` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.dropout(x, training=training)
