"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Test helper class LSTMModel."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: The input size (int) for   init  .
        hidden_size: The hidden size (int) for   init  .
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (output, (hn, cn)) = self.lstm(x)
    return output
