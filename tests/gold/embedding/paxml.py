"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import embedding_softmax


class EmbeddingModel(base_layer.BaseLayer):
  """Test helper class EmbeddingModel."""

  num_embeddings = 0
  embedding_dim = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "emb", embedding_softmax.Embedding.HParams(vocab_size=self.num_embeddings, embedding_dims=self.embedding_dim)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.emb(x)
