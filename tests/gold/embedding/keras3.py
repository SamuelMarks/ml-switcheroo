"""Test suite for the Keras3 module."""

import keras


class EmbeddingModel:
  """Test double `EmbeddingModel` implementing `call` for keras3 tests."""

  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Input `num_embeddings` parameter value (`int`).
        embedding_dim: Input `embedding_dim` parameter value (`int`).
    """
    super().__init__()
    self.emb = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `emb` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.emb(x)
