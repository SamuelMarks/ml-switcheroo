"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def causal_mask_fill(scores: jnp.ndarray, mask: jnp.ndarray, value: float = -1000000000.0) -> jnp.ndarray:
  """Helper to causal mask fill."""
  return jnp.where(mask == 0, value, scores)
