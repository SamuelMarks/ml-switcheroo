"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MLP:
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance.

    Args:
        in_features: Test argument or fixture.
        hidden_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.fc1 = tf.keras.layers.Dense(hidden_features, input_shape=(in_features,))
    self.fc2 = tf.keras.layers.Dense(out_features)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    x = self.fc1(x)
    x = tf.nn.relu(x)
    x = self.fc2(x)
    return x
