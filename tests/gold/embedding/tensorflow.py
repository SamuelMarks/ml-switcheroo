"""Test suite for the Tensorflow module."""

import tensorflow as tf


class EmbeddingModel:
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: The num embeddings (int) for   init  .
        embedding_dim: The embedding dim (int) for   init  .
    """
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.emb(x)
