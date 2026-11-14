"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Input `num_embeddings` parameter value (`int`).
        embedding_dim: Input `embedding_dim` parameter value (`int`).
    """
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `EmbeddingModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.emb(x)
