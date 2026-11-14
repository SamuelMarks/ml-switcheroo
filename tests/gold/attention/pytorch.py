"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
    """
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `AttentionModel` on input tensors.

    Args:
        query: Input `query` parameter value (`torch.Tensor`).
        key: Input `key` parameter value (`torch.Tensor`).
        value: Input `value` parameter value (`torch.Tensor`).

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (attn_output, _) = self.mha(query, key, value)
    return attn_output
