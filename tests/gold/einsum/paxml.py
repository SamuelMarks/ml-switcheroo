"""Test suite for the Paxml module."""

import jax.numpy as jnp


def bmm_einsum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jnp.einsum("bik,bkj->bij", x, y)
