"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class AttentionModel(nnx.Module):
  """Minimal `nnx.Module` test double used in flax NNX tests."""

  def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        query: Input `query` parameter value (`jnp.ndarray`).
        key: Input `key` parameter value (`jnp.ndarray`).
        value: Input `value` parameter value (`jnp.ndarray`).

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.mha(query, key, value)
