"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def clip_grads(model: nn.Module, max_norm: float = 1.0) -> None:
  """Helper to clip grads.

  Args:
      model: Test argument or fixture.
      max_norm: Test argument or fixture.
  """
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
