"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

import optax
from flax import nnx


def setup_adam(model: nnx.Module, lr: float = 0.001) -> jnp.ndarray:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `jnp.ndarray` from setup adam.
  """
  optimizer = nnx.Optimizer(model, optax.adam(lr))
  return optimizer
