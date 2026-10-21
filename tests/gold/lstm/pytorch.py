"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Test helper class LSTMModel."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Test argument or fixture.
        hidden_size: Test argument or fixture.
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    (output, (hn, cn)) = self.lstm(x)
    return output
