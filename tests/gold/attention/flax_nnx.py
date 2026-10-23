"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class AttentionModel(nnx.Module):
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        query: The query (jnp.ndarray) for   call  .
        key: The key (jnp.ndarray) for   call  .
        value: The value (jnp.ndarray) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.mha(query, key, value)
