"""Test suite for the Paxml module."""

import jax.numpy as jnp


def bmm_einsum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Helper to bmm einsum.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `jnp.ndarray` from bmm einsum.
  """
  return jnp.einsum("bik,bkj->bij", x, y)
