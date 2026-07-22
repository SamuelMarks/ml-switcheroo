"""Test suite for the Pytorch module."""

import torch


def conditional_op(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Helper to conditional op."""
  if pred.item():
    return x * 2
  else:
    return x + 2
