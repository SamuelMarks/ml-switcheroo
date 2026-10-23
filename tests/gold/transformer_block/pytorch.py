"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
  """Test helper class TransformerBlock."""

  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
    """Initializes the TransformerBlock instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
        ff_dim: The ff dim (int) for   init  .
        dropout: The dropout (float) for   init  .
    """
    super().__init__()
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout)
    )
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.
        attn_mask: The attn mask (torch.Tensor | None) for forward.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (attn_out, _) = self.attn(x, x, x, attn_mask=attn_mask)
    x = self.norm1(x + self.dropout(attn_out))
    ffn_out = self.ffn(x)
    x = self.norm2(x + ffn_out)
    return x
