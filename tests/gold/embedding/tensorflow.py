"""Test suite for the Tensorflow module."""

import tensorflow as tf


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Test argument or fixture.
        embedding_dim: Test argument or fixture.
    """
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.emb(x)
