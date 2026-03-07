"""Module docstring."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class WNModel(nn.Module):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PyTorch wraps the layer
    self.linear = weight_norm(nn.Linear(in_features, out_features))
    # </SWITCHEROO_FAILED_TO_TRANS>

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.linear(x)
