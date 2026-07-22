"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GAPModel(nn.Module):
  """Test suite for the G A P Model component."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return torch.mean(x, dim=(2, 3))
