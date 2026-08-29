"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class EmbeddingModel(nnx.Module):
  """Docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs):
    """Initializes the EmbeddingModel instance."""
    self.emb = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.emb(x)
