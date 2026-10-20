"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import attentions


class AttentionModel(base_layer.BaseLayer):  # type: ignore[misc]
  """Test helper class AttentionModel."""

  embed_dim = 0
  num_heads = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "mha",
      attentions.DotProductAttention.HParams(num_heads=self.num_heads, dim_per_head=self.embed_dim // self.num_heads),
    )

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.mha(query, key, value)
