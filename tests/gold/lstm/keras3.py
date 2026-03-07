"""Module docstring."""

import keras


class LSTMModel(keras.Model):
  """Class docstring."""

  def __init__(self, hidden_size: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Input shape usually inferred or passed to the layer directly
    self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x):
    """Function docstring."""
    return self.lstm(x)
