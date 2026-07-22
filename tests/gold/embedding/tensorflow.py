"""Test suite for the Tensorflow module."""

import tensorflow as tf


class EmbeddingModel(tf.keras.Model):
  """Test suite for the Embedding Model component."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance."""
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.emb(x)
