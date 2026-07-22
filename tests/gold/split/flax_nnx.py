"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, axis: int = -1):
  """Splits tensor."""
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
