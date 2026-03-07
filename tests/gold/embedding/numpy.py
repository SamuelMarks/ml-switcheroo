"""Module docstring."""

import numpy as np


class EmbeddingModel:
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.weight = np.random.randn(num_embeddings, embedding_dim)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Fancy indexing is supported in numpy
    return self.weight[x]
    # </SWITCHEROO_FAILED_TO_TRANS>
