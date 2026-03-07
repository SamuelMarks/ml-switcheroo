"""Module docstring."""

import torch
import torch.nn as nn


class GAPModel(nn.Module):
  """Class docstring."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return torch.mean(x, dim=(2, 3))
