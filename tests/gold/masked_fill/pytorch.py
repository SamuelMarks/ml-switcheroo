"""Test suite for the Pytorch module."""

import torch


def causal_mask_fill(scores: torch.Tensor, mask: torch.Tensor, value: float = -1000000000.0) -> torch.Tensor:
  """Helper to causal mask fill.

  Args:
      mask: Test argument or fixture.
      scores: Test argument or fixture.
      value: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return scores.masked_fill(mask == 0, value)
