"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import attentions
import jax.numpy as jnp


class AttentionModel(base_layer.BaseLayer):
  """Test suite for the Attention Model component."""

  embed_dim: int = 0
  num_heads: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child(
      "mha",
      attentions.DotProductAttention.HParams(num_heads=self.num_heads, dim_per_head=self.embed_dim // self.num_heads),
    )

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.mha(query, key, value)
