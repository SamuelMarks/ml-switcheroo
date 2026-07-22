"""Test suite for the Mlx module."""

import mlx.nn as nn
import mlx.optimizers as optim


def setup_adam(model: nn.Module, lr: float = 0.001):
  """Helper to setup adam."""
  return optim.Adam(learning_rate=lr)
