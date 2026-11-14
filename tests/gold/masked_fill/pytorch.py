"""Test suite for the Pytorch module."""

import torch


def causal_mask_fill(scores: torch.Tensor, mask: torch.Tensor, value: float = -1000000000.0) -> torch.Tensor:
  """Execute `scores.masked_fill` to support causal mask fill in pytorch tests.

  Args:
      mask: Input `mask` parameter value (`torch.Tensor`).
      scores: Input `scores` parameter value (`torch.Tensor`).
      value: Input `value` parameter value (`float`).

  Returns:
      Resulting `torch.Tensor` from causal mask fill.
  """
  return scores.masked_fill(mask == 0, value)
