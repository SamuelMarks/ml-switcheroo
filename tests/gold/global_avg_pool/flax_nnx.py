"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class GAPModel(nnx.Module):
  """Docstring."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return jnp.mean(x, axis=(1, 2))
