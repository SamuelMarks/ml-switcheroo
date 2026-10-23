"""Test suite for the Pytorch module."""

import torch


def concat_tensors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      dim: The dim (int) for concat tensors.

  Returns:
      Resulting `torch.Tensor` from concat tensors.
  """
  return torch.cat([x, y], dim=dim)
