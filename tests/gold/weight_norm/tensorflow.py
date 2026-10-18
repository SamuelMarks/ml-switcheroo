"""Test suite for the Tensorflow module."""

import tensorflow as tf
import tensorflow_addons as tfa


class WNModel:
  """Test helper class WNModel."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the WNModel instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = tfa.layers.WeightNormalization(tf.keras.layers.Dense(out_features))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
