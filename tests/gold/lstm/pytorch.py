"""Module docstring."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Class docstring."""

  def __init__(self, input_size: int, hidden_size: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    output, (hn, cn) = self.lstm(x)
    return output
