"""Module docstring."""

from flax import nnx
import jax.numpy as jnp
import jax


class TransformerBlock(nnx.Module):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, rngs: nnx.Rngs = None):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, dropout_rate=dropout, rngs=rngs)
    self.ffn_proj1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
    self.ffn_proj2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)
    self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout, rngs=rngs)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray = None, deterministic: bool = False) -> jnp.ndarray:
    """Function docstring."""
    attn_out = self.attn(x, x, x, mask=attn_mask, deterministic=deterministic)
    x = self.norm1(x + self.dropout(attn_out, deterministic=deterministic))
    ffn_out = self.ffn_proj1(x)
    ffn_out = jax.nn.gelu(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    ffn_out = self.ffn_proj2(ffn_out)
    ffn_out = self.dropout(ffn_out, deterministic=deterministic)
    x = self.norm2(x + ffn_out)
    return x
