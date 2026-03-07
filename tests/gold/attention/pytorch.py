"""Module docstring."""

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PyTorch defaults to sequence length first (batch_first=False) unless specified
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    attn_output, _ = self.mha(query, key, value)
    return attn_output
