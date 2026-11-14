"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp
from flax import nnx


class TransformerBlock(nnx.Module):
  """Minimal `nnx.Module` test double used in flax NNX tests."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
    rngs: nnx.Rngs | None = None,
  ):
    """Initializes the TransformerBlock instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
        ff_dim: Input `ff_dim` parameter value (`int`).
        dropout: Input `dropout` parameter value (`float`).
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, dropout_rate=dropout, rngs=rngs)
    self.ffn_proj1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
    self.ffn_proj2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)
    self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout, rngs=rngs)

  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray | None = None, deterministic: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        attn_mask: Input `attn_mask` parameter value (`jnp.ndarray | None`).
        deterministic: Input `deterministic` parameter value (`bool`).

    Returns:
        Output tensor or value computed by the forward pass.
    """
    attn_out = self.attn(x, x, x, mask=attn_mask, deterministic=deterministic)
    x = self.norm1(x + self.dropout(attn_out, deterministic=deterministic))
    ffn_out = self.ffn_proj1(x)
    ffn_out = jax.nn.gelu(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    ffn_out = self.ffn_proj2(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    return self.norm2(x + ffn_out)
