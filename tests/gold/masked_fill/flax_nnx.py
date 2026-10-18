"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def causal_mask_fill(scores: jnp.ndarray, mask: jnp.ndarray, value: float = -1000000000.0) -> jnp.ndarray:
  """Helper to causal mask fill.

  Args:
      mask: Test argument or fixture.
      scores: Test argument or fixture.
      value: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jnp.where(mask == 0, value, scores)
