"""Module docstring."""

import tensorflow as tf


class LSTMModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, hidden_size: int):
    """Function docstring."""
    super().__init__()
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.lstm(x)
