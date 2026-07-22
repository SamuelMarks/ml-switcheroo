"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
  """Test suite for the Attention Model component."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance."""
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    (attn_output, _) = self.mha(query, key, value)
    return attn_output
