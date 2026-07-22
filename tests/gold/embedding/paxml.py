"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import embedding_softmax
import jax.numpy as jnp


class EmbeddingModel(base_layer.BaseLayer):
  """Test suite for the Embedding Model component."""

  num_embeddings: int = 0
  embedding_dim: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child(
      "emb", embedding_softmax.Embedding.HParams(vocab_size=self.num_embeddings, embedding_dims=self.embedding_dim)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.emb(x)
