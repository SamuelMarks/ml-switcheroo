"""Test suite for the Paxml module."""

import jax.numpy as jnp


def bmm_einsum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Helper to bmm einsum."""
  return jnp.einsum("bik,bkj->bij", x, y)
