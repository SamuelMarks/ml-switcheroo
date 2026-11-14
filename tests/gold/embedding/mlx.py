"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class EmbeddingModel:
  """Sentinel `EmbeddingModel` fixture class used in MLX tests."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Input `num_embeddings` parameter value (`int`).
        embedding_dim: Input `embedding_dim` parameter value (`int`).
    """
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.emb(x)
