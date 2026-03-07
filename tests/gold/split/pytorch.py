"""Module docstring."""

import torch


def split_tensor(x: torch.Tensor, split_size: int, dim: int = -1):
  """Function docstring."""
  # PyTorch splits into chunks of split_size
  return torch.split(x, split_size, dim=dim)
