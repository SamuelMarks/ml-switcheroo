"""Test suite for the Keras3 module."""

import keras


class EmbeddingModel(keras.Model):
  """Test suite for the Embedding Model component."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance."""
    super().__init__()
    self.emb = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x):
    """Helper to call."""
    return self.emb(x)
