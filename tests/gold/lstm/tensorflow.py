"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LSTMModel:
  """Test helper class LSTMModel."""

  def __init__(self, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        hidden_size: Test argument or fixture.
    """
    super().__init__()
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.lstm(x)
