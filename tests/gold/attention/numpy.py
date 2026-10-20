"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
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
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    d_k = query.shape[-1]
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores / np.sum(scores, axis=-1, keepdims=True)
    return np.asarray(np.matmul(attn_weights, value), dtype=np.float64)
