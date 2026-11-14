"""Test suite for the Keras3 module."""

import keras


class LSTMModel:
  """Test double `LSTMModel` implementing `call` for keras3 tests."""

  def __init__(self, hidden_size: int) -> None:
    """Initializes the LSTMModel instance.

    Args:
        hidden_size: Input `hidden_size` parameter value (`int`).
    """
    super().__init__()
    self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `lstm` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.lstm(x)
