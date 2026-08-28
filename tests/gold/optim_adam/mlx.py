"""Test suite for the Mlx module."""

import typing
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore


def setup_adam(model: nn.Module, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam."""
  return optim.Adam(learning_rate=lr)
