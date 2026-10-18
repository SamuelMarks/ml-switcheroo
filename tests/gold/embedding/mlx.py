"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: typing.Any, embedding_dim: typing.Any):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
    """
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
