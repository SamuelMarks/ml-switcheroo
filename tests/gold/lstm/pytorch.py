"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Test suite for the L S T M Model component."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance."""
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    (output, (hn, cn)) = self.lstm(x)
    return output
