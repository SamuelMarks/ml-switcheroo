"""Test suite for the Pytorch module."""

import torch


def relu_activation(x: torch.Tensor) -> torch.Tensor:
  """Helper to relu activation."""
  return torch.relu(x)
