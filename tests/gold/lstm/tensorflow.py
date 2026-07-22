"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LSTMModel(tf.keras.Model):
  """Test suite for the L S T M Model component."""

  def __init__(self, hidden_size: int):
    """Initializes the LSTMModel instance."""
    super().__init__()
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.lstm(x)
