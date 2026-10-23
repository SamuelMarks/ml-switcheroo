"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
    """
    self.embed_dim = embed_dim
    self.num_heads = num_heads

  def __call__(
    self,
    query: np.typing.NDArray[np.float64],
    key: np.typing.NDArray[np.float64],
    value: np.typing.NDArray[np.float64],
  ) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        query: The query (np.typing.NDArray[np.float64]) for   call  .
        key: The key (np.typing.NDArray[np.float64]) for   call  .
        value: The value (np.typing.NDArray[np.float64]) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    d_k = query.shape[-1]
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores / np.sum(scores, axis=-1, keepdims=True)
    return np.asarray(np.matmul(attn_weights, value), dtype=np.float64)
