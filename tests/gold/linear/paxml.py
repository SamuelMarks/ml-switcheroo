"""Test suite for the Paxml module."""

import typing

from praxis.layers import linears


class Model:
  """Test helper class Model."""

  in_features: typing.Any = 0
  out_features: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child("linear", linears.Linear.HParams(input_dims=self.in_features, output_dims=self.out_features))

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
