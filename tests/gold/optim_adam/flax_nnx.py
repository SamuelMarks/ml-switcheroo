"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
import optax
from flax import nnx


def setup_adam(model: nnx.Module, lr: float = 0.001) -> jnp.ndarray:
  """Execute `nnx.Optimizer` to support setup adam in flax NNX tests.

  Args:
      model: Neural network model instance under test.
      lr: Input `lr` parameter value (`float`).

  Returns:
      Resulting `jnp.ndarray` from setup adam.
  """
  return nnx.Optimizer(model, optax.adam(lr))
