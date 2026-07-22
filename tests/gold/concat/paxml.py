"""Test suite for the Paxml module."""

import jax.numpy as jnp


def concat_tensors(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """Helper to concat tensors."""
  return jnp.concatenate([x, y], axis=axis)
