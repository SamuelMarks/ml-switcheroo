"""Test suite for the Numpy module."""

import typing

import numpy as np


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
    query: np.ndarray[typing.Any, typing.Any],
    key: np.ndarray[typing.Any, typing.Any],
    value: np.ndarray[typing.Any, typing.Any],
  ) -> typing.Any:
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
    return np.matmul(attn_weights, value)
