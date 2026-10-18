"""Test suite for the Pytorch module."""

import typing

import torch
import torch.nn as nn


class DropoutModel(nn.Module):
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def forward(self, x: torch.Tensor) -> typing.Any:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x)
