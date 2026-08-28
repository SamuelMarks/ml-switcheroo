"""Test suite for the Tensorflow module."""

import typing
import tensorflow as tf


class BNModel(tf.keras.Model):  # type: ignore
  """Test suite for the B N Model component."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance."""
    super().__init__()
    self.bn: typing.Any = tf.keras.layers.BatchNormalization()

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call."""
    return self.bn(x, training=training)
