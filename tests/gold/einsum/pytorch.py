"""Test suite for the Pytorch module."""

import torch


def bmm_einsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return torch.einsum("bik,bkj->bij", x, y)
