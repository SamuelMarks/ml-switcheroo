"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def setup_adam(model: nn.Module, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return optim.Adam(learning_rate=lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
