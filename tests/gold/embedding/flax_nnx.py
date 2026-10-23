"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class EmbeddingModel(nnx.Module):
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: The num embeddings (int) for   init  .
        embedding_dim: The embedding dim (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.emb = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.emb(x)
