"""TensorFlow golden reference for a Transformer encoder block."""

import tensorflow as tf


class TransformerBlock:
  """TensorFlow ``tf.keras`` Transformer encoder block."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
  ) -> None:
    """Initialize TensorFlow MultiHeadAttention, Dense FFN, and LayerNorm layers.

    Args:
        embed_dim: Model embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """
    self.attn = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads,
      key_dim=embed_dim // num_heads,
      dropout=dropout,
    )
    self.ffn = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(ff_dim, activation="gelu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(embed_dim),
        tf.keras.layers.Dropout(dropout),
      ]
    )
    self.norm1 = tf.keras.layers.LayerNormalization()
    self.norm2 = tf.keras.layers.LayerNormalization()
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, x: tf.Tensor, attn_mask: tf.Tensor | None = None) -> tf.Tensor:
    """Execute forward pass of the TensorFlow Transformer block.

    Args:
        x: Input sequence ``tf.Tensor``.
        attn_mask: Optional attention mask ``tf.Tensor``.

    Returns:
        Encoded ``tf.Tensor``.
    """
    attn_out = self.attn(x, x, x, attention_mask=attn_mask)
    h = self.norm1(x + self.dropout(attn_out))
    ffn_out = self.ffn(h)
    res: tf.Tensor = self.norm2(h + ffn_out)
    return res
