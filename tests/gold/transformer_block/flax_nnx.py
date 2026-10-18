"""Test suite for the Flax Nnx module."""

import typing

import jax
from flax import nnx


class TransformerBlock(nnx.Module):
  """Test helper class TransformerBlock."""

  def __init__(
    self,
    embed_dim: typing.Any,
    num_heads: typing.Any,
    ff_dim: typing.Any,
    dropout: typing.Any = 0.1,
    rngs: typing.Any = None,
  ):
    """Initializes the TransformerBlock instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
        ff_dim: Test argument or fixture.
        dropout: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, dropout_rate=dropout, rngs=rngs)
    self.ffn_proj1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
    self.ffn_proj2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)
    self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout, rngs=rngs)

  def __call__(self, x: typing.Any, attn_mask: typing.Any = None, deterministic: typing.Any = False) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        attn_mask: Test argument or fixture.
        deterministic: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    attn_out: typing.Any = self.attn(x, x, x, mask=attn_mask, deterministic=deterministic)
    x = self.norm1(x + self.dropout(attn_out, deterministic=deterministic))
    ffn_out: typing.Any = self.ffn_proj1(x)
    ffn_out = jax.nn.gelu(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    ffn_out = self.ffn_proj2(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    x = self.norm2(x + ffn_out)
    return x
