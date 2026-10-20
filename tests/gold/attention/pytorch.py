"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class AttentionModel(nn.Module):  # type: ignore[misc]
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
    """
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    (attn_output, _) = self.mha(query, key, value)
    return attn_output
