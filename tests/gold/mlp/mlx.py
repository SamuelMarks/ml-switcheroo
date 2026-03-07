"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, out_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    x = self.fc1(x)
    x = nn.relu(x)
    x = self.fc2(x)
    return x
