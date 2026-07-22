"""Test suite for the Pytorch module."""

import torch


def bmm_einsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Helper to bmm einsum."""
  return torch.einsum("bik,bkj->bij", x, y)
