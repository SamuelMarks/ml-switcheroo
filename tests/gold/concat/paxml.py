"""Test suite for the Paxml module."""

import jax.numpy as jnp


def concat_tensors(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """Execute `jnp.concatenate` to support concat tensors in PaxML tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `jnp.ndarray` from concat tensors.
  """
  return jnp.concatenate([x, y], axis=axis)
