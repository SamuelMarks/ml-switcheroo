"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class WNModel(nn.Module):  # type: ignore[misc]
  """Test helper class WNModel."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the WNModel instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = weight_norm(nn.Linear(in_features, out_features))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
