"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, axis: int = -1) -> jnp.ndarray:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: Input `split_size` parameter value (`int`).
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `jnp.ndarray` from split tensor.
  """
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
