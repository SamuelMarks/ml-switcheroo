"""Test suite for the Paxml module."""

import typing
from praxis import base_layer  # type: ignore
from praxis.layers import linears  # type: ignore
from praxis.layers import activations  # type: ignore


class MLP(base_layer.BaseLayer):  # type: ignore
  """Test suite for the M L P component."""

  in_features: int = 0
  hidden_features: int = 0
  out_features: int = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("fc1", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.hidden_features))
    self.create_child("relu", activations.ReLU.HParams())
    self.create_child("fc2", linears.Linear.HParams(input_dims=self.hidden_features, output_dims=self.out_features))

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
