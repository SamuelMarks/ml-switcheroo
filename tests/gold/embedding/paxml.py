"""Module docstring."""

from praxis import base_layer
from praxis.layers import embedding_softmax
import jax.numpy as jnp


class EmbeddingModel(base_layer.BaseLayer):
  """Class docstring."""

  num_embeddings: int = 0
  embedding_dim: int = 0

  def setup(self):
    """Function docstring."""
    self.create_child(
      "emb", embedding_softmax.Embedding.HParams(vocab_size=self.num_embeddings, embedding_dims=self.embedding_dim)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.emb(x)
