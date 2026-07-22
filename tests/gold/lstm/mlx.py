"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class LSTMModel(nn.Module):
  """Test suite for the L S T M Model component."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance."""
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    (output, (hn, cn)) = self.lstm(x)
    return output
