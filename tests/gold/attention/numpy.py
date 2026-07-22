"""Test suite for the Numpy module."""

import numpy as np


class AttentionModel:
  """Test suite for the Attention Model component."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance."""
    self.embed_dim = embed_dim
    self.num_heads = num_heads

  def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    d_k = query.shape[-1]
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores / np.sum(scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, value)
