"""Test suite for the Keras3 module."""

import typing

import keras


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
    """
    super().__init__()
    self.emb: typing.Any = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
