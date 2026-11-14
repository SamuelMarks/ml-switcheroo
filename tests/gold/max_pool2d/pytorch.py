"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MaxPoolModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Convolution or pooling spatial kernel size.
        stride: Convolution or pooling stride.
    """
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `MaxPoolModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.pool(x)
