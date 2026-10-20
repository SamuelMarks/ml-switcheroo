"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, axis: int = -1) -> jnp.ndarray:
  """Splits tensor.

  Args:
      x: Test argument or fixture.
      split_size: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
