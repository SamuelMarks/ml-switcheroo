"""Test suite for the Pytorch module."""

import typing

import torch.nn as nn
import torch.optim as optim


def setup_adam(model: nn.Module, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optim.Adam(model.parameters(), lr=lr)
