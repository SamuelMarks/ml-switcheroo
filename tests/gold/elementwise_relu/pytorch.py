"""Test suite for the Pytorch module."""

import torch


def relu_activation(x: torch.Tensor) -> torch.Tensor:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return torch.relu(x)
