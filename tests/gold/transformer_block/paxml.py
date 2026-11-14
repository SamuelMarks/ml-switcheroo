"""PaxML / Praxis golden reference for a Transformer encoder block."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import attentions, linears, normalizations


class TransformerBlock(base_layer.BaseLayer):
  """Praxis Transformer encoder block with multi-head attention and FFN."""

  embed_dim = 64
  num_heads = 4
  ff_dim = 256
  dropout_rate = 0.1

  def setup(self) -> None:
    """Create Praxis attention, layer norm, and feed-forward child layers."""
    self.create_child(
      "attn",
      attentions.DotProductAttention.HParams(
        input_dim=self.embed_dim,
        hidden_dim=self.embed_dim,
        num_heads=self.num_heads,
      ),
    )
    self.create_child("ff1", linears.Linear.HParams(input_dims=self.embed_dim, output_dims=self.ff_dim))
    self.create_child("ff2", linears.Linear.HParams(input_dims=self.ff_dim, output_dims=self.embed_dim))
    self.create_child("norm1", normalizations.LayerNorm.HParams(dim=self.embed_dim))
    self.create_child("norm2", normalizations.LayerNorm.HParams(dim=self.embed_dim))

  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Apply Praxis attention and feed-forward sublayers.

    Args:
        x: Input sequence array.
        attn_mask: Optional attention bias or mask array.

    Returns:
        Encoded JAX sequence array.
    """
    attn_out, _ = self.attn(x, x, x, atten_mask=attn_mask)
    h = self.norm1(x + attn_out)
    ff_out = self.ff2(jnp.maximum(0.0, self.ff1(h)))
    res: jnp.ndarray = self.norm2(h + ff_out)
    return res
