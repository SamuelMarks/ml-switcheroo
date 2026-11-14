"""Test suite for the Pytorch module."""

import torch


def conditional_op(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Execute `pred.item` to support conditional operation in pytorch tests.

  Args:
      pred: Input `pred` parameter value (`torch.Tensor`).
      x: Primary input tensor or array operand.

  Returns:
      Resulting `torch.Tensor` from conditional op.
  """
  if pred.item():
    return x * 2
  return x + 2
