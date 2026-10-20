"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class SameConvModel(nn.Module):  # type: ignore[misc]
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
