"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class Model(nn.Module):
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
