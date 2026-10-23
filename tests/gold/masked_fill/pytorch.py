"""Test suite for the Pytorch module."""

import torch


def causal_mask_fill(scores: torch.Tensor, mask: torch.Tensor, value: float = -1000000000.0) -> torch.Tensor:
  """Helper to causal mask fill.

  Args:
      mask: The mask (torch.Tensor) for causal mask fill.
      scores: The scores (torch.Tensor) for causal mask fill.
      value: The value (float) for causal mask fill.

  Returns:
      Resulting `torch.Tensor` from causal mask fill.
  """
  return scores.masked_fill(mask == 0, value)
