"""Module docstring."""

import tensorflow as tf


class EmbeddingModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Function docstring."""
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.emb(x)
