"""Test suite for the Paxml module."""

import typing
from praxis import optimizers  # type: ignore


def setup_adam(model: typing.Any, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam."""
  return optimizers.Adam.HParams(learning_rate=lr)
