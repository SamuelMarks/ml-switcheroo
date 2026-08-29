"""Test suite for the Keras3 module."""

import typing

import keras


class EmbeddingModel(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    """Initializes the EmbeddingModel instance."""
    super().__init__()
    self.emb: typing.Any = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.emb(x)
