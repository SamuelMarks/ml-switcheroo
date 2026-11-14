"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class SameConvModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Number of input convolution channels.
        out_channels: Number of output convolution channels.
        kernel_size: Convolution or pooling spatial kernel size.
    """
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `SameConvModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.conv(x)
