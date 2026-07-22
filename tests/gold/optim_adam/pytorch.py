"""Test suite for the Pytorch module."""

import torch.nn as nn
import torch.optim as optim


def setup_adam(model: nn.Module, lr: float = 0.001):
  """Helper to setup adam."""
  return optim.Adam(model.parameters(), lr=lr)
