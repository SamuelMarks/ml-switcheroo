"""Test suite for the Mlx module."""

import mlx.nn as nn


def create_sequential(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
  """Creates sequential."""
  return nn.Sequential(nn.Linear(in_features, hidden), nn.relu, nn.Linear(hidden, out_features))
