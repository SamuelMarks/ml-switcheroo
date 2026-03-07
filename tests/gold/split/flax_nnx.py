"""Module docstring."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, axis: int = -1):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX jnp.split expects the NUMBER of arrays or indices, not chunk sizes.
  # Usually requires arithmetic: x.shape[axis] // split_size
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
  # </SWITCHEROO_FAILED_TO_TRANS>
