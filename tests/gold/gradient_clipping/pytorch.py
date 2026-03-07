"""Module docstring."""

import torch
import torch.nn as nn


def clip_grads(model: nn.Module, max_norm: float = 1.0):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # In-place clipping in PyTorch
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
  # </SWITCHEROO_FAILED_TO_TRANS>
