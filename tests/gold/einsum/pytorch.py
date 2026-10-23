"""Test suite for the Pytorch module."""

import torch


def bmm_einsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Helper to bmm einsum.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `torch.Tensor` from bmm einsum.
  """
  return torch.einsum("bik,bkj->bij", x, y)
