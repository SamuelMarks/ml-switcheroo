"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def concat_tensors(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: The axis (int) for concat tensors.

  Returns:
      Resulting `jnp.ndarray` from concat tensors.
  """
  return jnp.concatenate([x, y], axis=axis)
