"""Test suite for the Paxml module."""

import typing

from praxis.layers import normalizations


class LayerNormModel:
  """Test helper class LayerNormModel."""

  normalized_shape: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child("ln", normalizations.LayerNorm.HParams(dim=self.normalized_shape))

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
