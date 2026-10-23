"""Test suite for the Pytorch module."""

import torch


def relu_activation(x: torch.Tensor) -> torch.Tensor:
  """Helper to relu activation.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `torch.Tensor` from relu activation.
  """
  return torch.relu(x)
