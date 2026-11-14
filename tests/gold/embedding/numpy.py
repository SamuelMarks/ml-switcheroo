"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class EmbeddingModel:
  """Sentinel `EmbeddingModel` fixture class used in numpy tests."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: Input `num_embeddings` parameter value (`int`).
        embedding_dim: Input `embedding_dim` parameter value (`int`).
    """
    self.weight = np.random.randn(num_embeddings, embedding_dim)

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return np.asarray(self.weight[x], dtype=np.float64)
