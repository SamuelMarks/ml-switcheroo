"""Test suite for the Pytorch module."""

import typing

import torch


def split_tensor(x: torch.Tensor, split_size: int, dim: int = -1) -> typing.Any:
  """Splits tensor.

  Args:
      x: Test argument or fixture.
      split_size: Test argument or fixture.
      dim: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return torch.split(x, split_size, dim=dim)
