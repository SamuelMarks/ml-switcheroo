"""Module docstring."""

import jax.numpy as jnp


def bmm_einsum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  # JAX
  return jnp.einsum("bik,bkj->bij", x, y)
