"""Test suite for the Pytorch module."""

import torch


def relu_activation(x: torch.Tensor) -> torch.Tensor:
  """Execute `torch.relu` to support relu activation in pytorch tests.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `torch.Tensor` from relu activation.
  """
  return torch.relu(x)
