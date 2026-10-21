"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MaxPoolModel(nn.Module):
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Test argument or fixture.
        stride: Test argument or fixture.
    """
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.pool(x)
