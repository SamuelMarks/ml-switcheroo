"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

import optax
from flax import nnx


def setup_adam(model: nnx.Module, lr: float = 0.001) -> jnp.ndarray:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  optimizer = nnx.Optimizer(model, optax.adam(lr))
  return optimizer
