"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LSTMModel:
  """Test helper class LSTMModel."""

  def __init__(self, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        hidden_size: The hidden size (int) for   init  .
    """
    super().__init__()
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.lstm(x)
