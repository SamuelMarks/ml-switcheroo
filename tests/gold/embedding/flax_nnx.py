"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class EmbeddingModel(nnx.Module):
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.emb = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
