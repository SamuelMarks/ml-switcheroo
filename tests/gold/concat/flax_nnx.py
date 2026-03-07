"""Module docstring."""

import jax.numpy as jnp


def concat_tensors(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """Function docstring."""
  # JAX uses 'axis'
  return jnp.concatenate([x, y], axis=axis)
