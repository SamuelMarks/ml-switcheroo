"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
    """
    super().__init__()
    self.bn: typing.Any = tf.keras.layers.BatchNormalization()

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x, training=training)
