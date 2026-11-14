"""Test suite for the Tensorflow module."""

import tensorflow as tf


class EmbeddingModel:
  """Test double `EmbeddingModel` implementing `call` for tensorflow tests."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Input `num_embeddings` parameter value (`int`).
        embedding_dim: Input `embedding_dim` parameter value (`int`).
    """
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `emb` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.emb(x)
