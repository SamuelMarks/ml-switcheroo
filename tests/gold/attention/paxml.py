"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import attentions


class AttentionModel(base_layer.BaseLayer):
  """Docstring."""

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
