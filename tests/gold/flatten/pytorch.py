"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class FlattenModel(nn.Module):  # type: ignore[misc]
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Test argument or fixture.
    """
    super().__init__()
    self.flatten = nn.Flatten(start_dim=start_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.flatten(x)
