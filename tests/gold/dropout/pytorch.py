"""Module docstring."""

import torch
import torch.nn as nn


class DropoutModel(nn.Module):
  """Class docstring."""

  def __init__(self, p: float = 0.5):
    """Function docstring."""
    super().__init__()
    self.dropout = nn.Dropout(p)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PyTorch relies on the global self.training flag
    return self.dropout(x)
    # </SWITCHEROO_FAILED_TO_TRANS>
