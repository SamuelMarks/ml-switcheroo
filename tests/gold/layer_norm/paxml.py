"""Test suite for the Paxml module."""

import typing

from praxis import base_layer  # type: ignore
from praxis.layers import normalizations  # type: ignore


class LayerNormModel(base_layer.BaseLayer):  # type: ignore
  """Docstring."""

  normalized_shape: int = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("ln", normalizations.LayerNorm.HParams(dim=self.normalized_shape))

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    return self.ln(x)
