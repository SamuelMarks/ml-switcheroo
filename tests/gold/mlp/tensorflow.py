"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MLP(tf.keras.Model):
  """Test suite for the M L P component."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance."""
    super().__init__()
    self.fc1 = tf.keras.layers.Dense(hidden_features, input_shape=(in_features,))
    self.fc2 = tf.keras.layers.Dense(out_features)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    x = self.fc1(x)
    x = tf.nn.relu(x)
    x = self.fc2(x)
    return x
