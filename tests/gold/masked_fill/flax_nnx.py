"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def causal_mask_fill(scores: jnp.ndarray, mask: jnp.ndarray, value: float = -1000000000.0) -> jnp.ndarray:
  """Helper to causal mask fill.

  Args:
      mask: The mask (jnp.ndarray) for causal mask fill.
      scores: The scores (jnp.ndarray) for causal mask fill.
      value: The value (float) for causal mask fill.

  Returns:
      Resulting `jnp.ndarray` from causal mask fill.
  """
  return jnp.where(mask == 0, value, scores)
