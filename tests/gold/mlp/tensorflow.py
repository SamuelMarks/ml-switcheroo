"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MLP:
  """Test double `MLP` implementing `call` for tensorflow tests."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: Input `hidden_features` parameter value (`int`).
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = tf.keras.layers.Dense(hidden_features, input_shape=(in_features,))
    self.fc2 = tf.keras.layers.Dense(out_features)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `fc1` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    x = self.fc1(x)
    x = tf.nn.relu(x)
    return self.fc2(x)
