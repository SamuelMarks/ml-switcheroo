"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, axis: int = -1) -> jnp.ndarray:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: The split size (int) for split tensor.
      axis: The axis (int) for split tensor.

  Returns:
      Resulting `jnp.ndarray` from split tensor.
  """
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
