"""Test suite for the Pytorch module."""

import torch


def bmm_einsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Execute `torch.einsum` to support bmm einsum in pytorch tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `torch.Tensor` from bmm einsum.
  """
  return torch.einsum("bik,bkj->bij", x, y)
