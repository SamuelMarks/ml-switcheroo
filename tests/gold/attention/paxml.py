"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import attentions


class AttentionModel(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  embed_dim = 0
  num_heads = 0

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
    self.create_child(
      "mha",
      attentions.DotProductAttention.HParams(num_heads=self.num_heads, dim_per_head=self.embed_dim // self.num_heads),
    )

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
