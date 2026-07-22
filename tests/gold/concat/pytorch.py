"""Test suite for the Pytorch module."""

import torch


def concat_tensors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Helper to concat tensors."""
  return torch.cat([x, y], dim=dim)
