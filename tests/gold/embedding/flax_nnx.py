"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class EmbeddingModel(nnx.Module):
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.emb = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.emb(x)
