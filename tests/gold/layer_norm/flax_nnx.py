"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LayerNormModel(nnx.Module):
  """Docstring."""

  def __init__(self, normalized_shape: int, rngs: nnx.Rngs):
    """Initializes the LayerNormModel instance."""
    self.ln = nnx.LayerNorm(normalized_shape, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.ln(x)
