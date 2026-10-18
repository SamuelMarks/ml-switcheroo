"""Test suite for the Paxml module."""

import typing

from praxis.layers import normalizations


class BNModel:
  """Test helper class BNModel."""

  num_features: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child("bn", normalizations.BatchNorm.HParams(dim=self.num_features))

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x)
