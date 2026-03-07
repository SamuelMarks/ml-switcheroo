"""Module docstring."""

import tensorflow as tf


class MLP(tf.keras.Model):
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.fc1 = tf.keras.layers.Dense(hidden_features, input_shape=(in_features,))
    self.fc2 = tf.keras.layers.Dense(out_features)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    x = self.fc1(x)
    x = tf.nn.relu(x)
    x = self.fc2(x)
    return x
