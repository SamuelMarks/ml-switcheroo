"""Test suite for the Paxml module."""

import typing

from praxis import base_layer  # type: ignore
from praxis.layers import stochastic  # type: ignore


class DropoutModel(base_layer.BaseLayer):  # type: ignore
  """Docstring."""

  p: float = 0.5

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child("dropout", stochastic.Dropout.HParams(keep_prob=1.0 - self.p))

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    return self.dropout(x)
