"""Module docstring."""

from praxis import base_layer
from praxis.layers import attentions
import jax.numpy as jnp


class AttentionModel(base_layer.BaseLayer):
  """Class docstring."""

  embed_dim: int = 0
  num_heads: int = 0

  def setup(self):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.create_child(
      "mha",
      attentions.DotProductAttention.HParams(num_heads=self.num_heads, dim_per_head=self.embed_dim // self.num_heads),
    )
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.mha(query, key, value)
