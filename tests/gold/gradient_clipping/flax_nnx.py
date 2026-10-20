"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

import optax


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> jnp.ndarray:
  """Gets clipped optimizer.

  Args:
      learning_rate: Test argument or fixture.
      max_norm: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
