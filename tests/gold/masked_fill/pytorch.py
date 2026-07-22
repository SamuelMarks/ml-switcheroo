"""Test suite for the Pytorch module."""

import torch


def causal_mask_fill(scores: torch.Tensor, mask: torch.Tensor, value: float = -1000000000.0) -> torch.Tensor:
  """Helper to causal mask fill."""
  return scores.masked_fill(mask == 0, value)
