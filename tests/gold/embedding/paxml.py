"""Test suite for the Paxml module."""

import typing

from praxis import base_layer  # type: ignore
from praxis.layers import embedding_softmax  # type: ignore


class EmbeddingModel(base_layer.BaseLayer):  # type: ignore
  """Docstring."""

  num_embeddings: int = 0
  embedding_dim: int = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "emb", embedding_softmax.Embedding.HParams(vocab_size=self.num_embeddings, embedding_dims=self.embedding_dim)
    )

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    return self.emb(x)
