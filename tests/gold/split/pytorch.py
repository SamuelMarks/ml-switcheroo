"""Test suite for the Pytorch module."""

import torch


def split_tensor(x: torch.Tensor, split_size: int, dim: int = -1) -> torch.Tensor:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: The split size (int) for split tensor.
      dim: The dim (int) for split tensor.

  Returns:
      Resulting `torch.Tensor` from split tensor.
  """
  return torch.split(x, split_size, dim=dim)
