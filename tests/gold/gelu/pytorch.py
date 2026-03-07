"""Module docstring."""

import torch
import torch.nn as nn


def gelu_activation(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
  """Function docstring."""
  # PyTorch uses string 'none' or 'tanh'
  return nn.functional.gelu(x, approximate=approximate)
