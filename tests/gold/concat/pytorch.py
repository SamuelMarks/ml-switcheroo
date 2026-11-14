"""Test suite for the Pytorch module."""

import torch


def concat_tensors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Execute `torch.cat` to support concat tensors in pytorch tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      dim: Input `dim` parameter value (`int`).

  Returns:
      Resulting `torch.Tensor` from concat tensors.
  """
  return torch.cat([x, y], dim=dim)
