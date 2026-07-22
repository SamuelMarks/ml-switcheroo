"""Test suite for the Flax Nnx module."""

from flax import nnx
import optax


def setup_adam(model: nnx.Module, lr: float = 0.001):
  """Helper to setup adam."""
  optimizer = nnx.Optimizer(model, optax.adam(lr))
  return optimizer
