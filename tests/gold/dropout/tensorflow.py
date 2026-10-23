"""Test suite for the Tensorflow module."""

import tensorflow as tf


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
    """
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(p)

  def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool | None) for call.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.dropout(x, training=training)
