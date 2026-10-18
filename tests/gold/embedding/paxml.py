"""Test suite for the Paxml module."""

import typing

from praxis.layers import embedding_softmax


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  num_embeddings: typing.Any = 0
  embedding_dim: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child(
      "emb", embedding_softmax.Embedding.HParams(vocab_size=self.num_embeddings, embedding_dims=self.embedding_dim)
    )

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
