"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
    """
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        query: The query (torch.Tensor) for forward.
        key: The key (torch.Tensor) for forward.
        value: The value (torch.Tensor) for forward.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (attn_output, _) = self.mha(query, key, value)
    return attn_output
