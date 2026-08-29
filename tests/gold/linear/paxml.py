"""Test suite for the Paxml module."""

import typing

from praxis import base_layer  # type: ignore
from praxis.layers import linears  # type: ignore


class Model(base_layer.BaseLayer):  # type: ignore
  """Docstring."""

  in_features: int = 0
  out_features: int = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("linear", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.out_features))

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    return self.linear(x)
