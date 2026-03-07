"""Module docstring."""

import torch


def conditional_op(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PyTorch allows standard Python if/else
  if pred.item():
    return x * 2
  else:
    return x + 2
  # </SWITCHEROO_FAILED_TO_TRANS>
