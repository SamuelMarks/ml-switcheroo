"""Module docstring."""

import jax.numpy as jnp


def causal_mask_fill(scores: jnp.ndarray, mask: jnp.ndarray, value: float = -1e9) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX uses jnp.where
  return jnp.where(mask == 0, value, scores)
  # </SWITCHEROO_FAILED_TO_TRANS>
