"""Test suite for the Numpy module."""

import typing

import numpy as np


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
    """
    self.weight = np.random.randn(num_embeddings, embedding_dim)

  def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.weight[x]
