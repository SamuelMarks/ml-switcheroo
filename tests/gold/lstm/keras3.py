"""Test suite for the Keras3 module."""

import keras


class LSTMModel:
  """Test helper class LSTMModel."""

  def __init__(self, hidden_size: int) -> None:
    """Initializes the LSTMModel instance.

    Args:
        hidden_size: The hidden size (int) for   init  .
    """
    super().__init__()
    self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.lstm(x)
