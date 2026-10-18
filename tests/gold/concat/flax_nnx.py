"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def concat_tensors(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jnp.concatenate([x, y], axis=axis)
