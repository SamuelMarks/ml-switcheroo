"""Test suite for the Keras3 module."""

import typing
import keras


class LSTMModel(keras.Model):  # type: ignore
  """Test suite for the L S T M Model component."""

  def __init__(self, hidden_size: int) -> None:
    """Initializes the LSTMModel instance."""
    super().__init__()
    self.lstm: typing.Any = keras.layers.LSTM(hidden_size, return_sequences=True)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.lstm(x)
