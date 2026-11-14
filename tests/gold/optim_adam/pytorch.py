"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
import torch.optim as optim


def setup_adam(model: nn.Module, lr: float = 0.001) -> torch.Tensor:
  """Execute `optim.Adam` to support setup adam in pytorch tests.

  Args:
      model: Neural network model instance under test.
      lr: Input `lr` parameter value (`float`).

  Returns:
      Resulting `torch.Tensor` from setup adam.
  """
  return optim.Adam(model.parameters(), lr=lr)
