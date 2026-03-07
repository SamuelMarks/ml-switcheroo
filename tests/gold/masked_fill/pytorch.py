"""Module docstring."""

import torch


def causal_mask_fill(scores: torch.Tensor, mask: torch.Tensor, value: float = -1e9) -> torch.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PyTorch masked_fill_ modifies a copy or in-place based on usage
  return scores.masked_fill(mask == 0, value)
  # </SWITCHEROO_FAILED_TO_TRANS>
