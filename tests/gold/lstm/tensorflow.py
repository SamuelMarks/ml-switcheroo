"""Test suite for the Tensorflow module."""

import tensorflow as tf


class LSTMModel:
  """Test double `LSTMModel` implementing `call` for tensorflow tests."""

  def __init__(self, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        hidden_size: Input `hidden_size` parameter value (`int`).
    """
    super().__init__()
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `lstm` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.lstm(x)
