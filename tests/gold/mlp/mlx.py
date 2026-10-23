"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class MLP:
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: The hidden features (int) for   init  .
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, out_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    x = self.fc1(x)
    x = nn.relu(x)
    x = self.fc2(x)
    return x
