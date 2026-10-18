"""Test suite for the Pytorch module."""

import torch


def conditional_op(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Helper to conditional op.

  Args:
      pred: Test argument or fixture.
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  if pred.item():
    return x * 2
  else:
    return x + 2
