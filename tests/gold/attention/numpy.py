"""Module docstring."""

import numpy as np


class AttentionModel:
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Function docstring."""
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Manual projection matrices omitted for brevity
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Simplified scaled dot-product attention
    d_k = query.shape[-1]
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
    # Numerically stable softmax
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores / np.sum(scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, value)
    # </SWITCHEROO_FAILED_TO_TRANS>
