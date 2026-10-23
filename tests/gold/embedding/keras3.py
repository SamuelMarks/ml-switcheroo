"""Test suite for the Keras3 module."""

import keras


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: The num embeddings (int) for   init  .
        embedding_dim: The embedding dim (int) for   init  .
    """
    super().__init__()
    self.emb = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.emb(x)
