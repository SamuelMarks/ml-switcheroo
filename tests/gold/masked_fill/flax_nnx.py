"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def causal_mask_fill(scores: jnp.ndarray, mask: jnp.ndarray, value: float = -1000000000.0) -> jnp.ndarray:
  """Execute `jnp.where` to support causal mask fill in flax NNX tests.

  Args:
      mask: Input `mask` parameter value (`jnp.ndarray`).
      scores: Input `scores` parameter value (`jnp.ndarray`).
      value: Input `value` parameter value (`float`).

  Returns:
      Resulting `jnp.ndarray` from causal mask fill.
  """
  return jnp.where(mask == 0, value, scores)
