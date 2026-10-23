"""Test suite for the Tensorflow module."""

import tensorflow as tf


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    super().__init__()
    self.linear = tf.keras.layers.Dense(out_features, input_shape=(in_features,))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.linear(x)
