"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
import torch.optim as optim


def setup_adam(model: nn.Module, lr: float = 0.001) -> torch.Tensor:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `torch.Tensor` from setup adam.
  """
  return optim.Adam(model.parameters(), lr=lr)
