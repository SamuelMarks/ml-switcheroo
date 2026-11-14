"""Test suite for the Pytorch module."""

import torch


def split_tensor(x: torch.Tensor, split_size: int, dim: int = -1) -> torch.Tensor:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: Input `split_size` parameter value (`int`).
      dim: Input `dim` parameter value (`int`).

  Returns:
      Resulting `torch.Tensor` from split tensor.
  """
  return torch.split(x, split_size, dim=dim)
