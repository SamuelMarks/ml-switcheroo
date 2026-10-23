"""Test suite for the Pytorch module."""

import torch


def conditional_op(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Helper to conditional op.

  Args:
      pred: The pred (torch.Tensor) for conditional op.
      x: Primary input tensor or array operand.

  Returns:
      Resulting `torch.Tensor` from conditional op.
  """
  if pred.item():
    return x * 2
  else:
    return x + 2
