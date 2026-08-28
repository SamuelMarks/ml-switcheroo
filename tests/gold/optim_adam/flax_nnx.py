"""Test suite for the Flax Nnx module."""

import typing
from flax import nnx  # type: ignore
import optax  # type: ignore


def setup_adam(model: nnx.Module, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam."""
  optimizer = nnx.Optimizer(model, optax.adam(lr))
  return optimizer
