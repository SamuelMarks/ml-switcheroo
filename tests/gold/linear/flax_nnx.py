"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class Model(nnx.Module):
  """Docstring."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the Model instance."""
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.linear(x)
