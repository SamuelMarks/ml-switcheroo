"""Module docstring."""

import torch


def bmm_einsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Function docstring."""
  # PyTorch
  return torch.einsum("bik,bkj->bij", x, y)
