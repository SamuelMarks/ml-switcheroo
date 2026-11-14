"""Apple MLX golden reference for a Transformer encoder block."""

import mlx.core as mx
import mlx.nn as nn


class TransformerBlock:
  """Post-norm Transformer encoder block in Apple MLX."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
  ) -> None:
    """Initialize self-attention, feed-forward network, norms, and dropout.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Hidden dimension of the feed-forward sublayer.
        dropout: Dropout probability.
    """
    self.attn = nn.MultiHeadAttention(embed_dim, num_heads)
    self.linear1 = nn.Linear(embed_dim, ff_dim)
    self.gelu = nn.GELU()
    self.linear2 = nn.Linear(ff_dim, embed_dim)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def __call__(self, x: mx.array, attn_mask: mx.array | None = None) -> mx.array:
    """Apply self-attention and feed-forward residual sublayers to ``x``.

    Args:
        x: Input sequence array of shape ``(B, T, D)``.
        attn_mask: Optional additive or boolean attention mask.

    Returns:
        Encoded sequence array of shape ``(B, T, D)``.
    """
    attn_out = self.attn(x, x, x, mask=attn_mask)
    x = self.norm1(x + self.dropout(attn_out))
    ffn_hidden = self.dropout(self.gelu(self.linear1(x)))
    ffn_out = self.dropout(self.linear2(ffn_hidden))
    res: mx.array = self.norm2(x + ffn_out)
    return res
