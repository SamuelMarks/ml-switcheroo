"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class MLP:
  """Test helper class MLP."""

  def __init__(self, in_features: typing.Any, hidden_features: typing.Any, out_features: typing.Any):
    """Initializes the MLP instance.

    Args:
        in_features: Test argument or fixture.
        hidden_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, out_features)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    x = self.fc1(x)
    x = nn.relu(x)
    x = self.fc2(x)
    return x
