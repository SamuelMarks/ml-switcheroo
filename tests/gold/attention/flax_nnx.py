"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class AttentionModel(nnx.Module):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # NNX specifies the number of heads and features explicitly
    self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # NNX computes multihead attention as a single call typically expecting inputs as (query, key, value)
    return self.mha(query, key, value)
