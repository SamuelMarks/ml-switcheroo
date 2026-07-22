"""Test suite for the Pytorch module."""

import torch


def split_tensor(x: torch.Tensor, split_size: int, dim: int = -1):
  """Splits tensor."""
  return torch.split(x, split_size, dim=dim)
