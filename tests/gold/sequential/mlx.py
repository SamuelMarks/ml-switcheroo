"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


def create_sequential(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
  """Function docstring."""
  return nn.Sequential(nn.Linear(in_features, hidden), nn.relu, nn.Linear(hidden, out_features))
