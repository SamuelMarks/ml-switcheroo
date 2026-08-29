"""Test suite for the Numpy module."""

import numpy as np


class EmbeddingModel:
  """Docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance."""
    self.weight = np.random.randn(num_embeddings, embedding_dim)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    return self.weight[x]
