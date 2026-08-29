"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class AttentionModel(nnx.Module):
  """Docstring."""

  def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs):
    """Initializes the AttentionModel instance."""
    self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.mha(query, key, value)
