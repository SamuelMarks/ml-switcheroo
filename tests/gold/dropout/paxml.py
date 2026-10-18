"""Test suite for the Paxml module."""

import typing

from praxis.layers import stochastic


class DropoutModel:
  """Test helper class DropoutModel."""

  p: typing.Any = 0.5

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x)
