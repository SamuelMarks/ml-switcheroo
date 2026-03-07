"""Module docstring."""

import keras


class EmbeddingModel(keras.Model):
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Function docstring."""
    super().__init__()
    self.emb = keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)

  def call(self, x):
    """Function docstring."""
    return self.emb(x)
