"""Test suite for the Keras3 module."""

import keras


class LSTMModel(keras.Model):
  """Test suite for the L S T M Model component."""

  def __init__(self, hidden_size: int):
    """Initializes the LSTMModel instance."""
    super().__init__()
    self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x):
    """Helper to call."""
    return self.lstm(x)
