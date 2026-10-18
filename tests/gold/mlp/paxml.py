"""Test suite for the Paxml module."""

import typing

from praxis.layers import activations
from praxis.layers import linears


class MLP:
  """Test helper class MLP."""

  in_features: typing.Any = 0
  hidden_features: typing.Any = 0
  out_features: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child("fc1", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.hidden_features))
    self.create_child("relu", activations.ReLU.HParams())
    self.create_child("fc2", linears.Linear.HParams(input_dims=self.hidden_features, output_dims=self.out_features))

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
