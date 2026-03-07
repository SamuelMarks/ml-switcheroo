"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class LSTMModel(nn.Module):
  """Class docstring."""

  def __init__(self, input_size: int, hidden_size: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # MLX may not have a fully unified LSTM layer exactly matching PyTorch yet, but we use the RNN primitives
    self.lstm = nn.LSTM(input_size, hidden_size)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    # returns output, (hn, cn) usually if implemented fully, or just processes seq
    output, (hn, cn) = self.lstm(x)
    return output
