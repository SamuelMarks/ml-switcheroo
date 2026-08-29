"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class DropoutModel(nnx.Module):
  """Docstring."""

  def __init__(self, p: float = 0.5, rngs: nnx.Rngs = None):
    """Initializes the DropoutModel instance."""
    self.dropout = nnx.Dropout(p, rngs=rngs)

  def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.dropout(x, deterministic=deterministic)
