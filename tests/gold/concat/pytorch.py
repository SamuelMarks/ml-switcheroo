"""Test suite for the Pytorch module."""

import torch


def concat_tensors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      dim: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return torch.cat([x, y], dim=dim)
