"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GAPModel(nn.Module):  # type: ignore[misc]
  """Test helper class GAPModel."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return torch.mean(x, dim=(2, 3))
