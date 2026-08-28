"""Test suite for the Flax Nnx module."""

import typing
import jax.numpy as jnp


def split_tensor(x: typing.Any, split_size: int, axis: int = -1) -> typing.Any:
  """Splits tensor."""
  num_splits = x.shape[axis] // split_size
  return jnp.split(x, num_splits, axis=axis)
