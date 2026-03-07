"""Module docstring."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout)
    )
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    """Function docstring."""
    attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
    x = self.norm1(x + self.dropout(attn_out))
    ffn_out = self.ffn(x)
    x = self.norm2(x + ffn_out)
    return x
