"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Input `input_size` parameter value (`int`).
        hidden_size: Input `hidden_size` parameter value (`int`).
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `LSTMModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (output, (_hn, _cn)) = self.lstm(x)
    return output
