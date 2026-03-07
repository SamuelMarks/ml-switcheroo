"""Module docstring."""

import torch


def concat_tensors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Function docstring."""
  # PyTorch uses 'dim'
  return torch.cat([x, y], dim=dim)
