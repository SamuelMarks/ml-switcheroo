"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
    """
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
