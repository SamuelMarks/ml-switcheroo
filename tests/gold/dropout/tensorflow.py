"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(p)

  def call(self, x: tf.Tensor, training: typing.Optional[bool] = None) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x, training=training)
