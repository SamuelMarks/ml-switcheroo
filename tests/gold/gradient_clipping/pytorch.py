"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def clip_grads(model: nn.Module, max_norm: float = 1.0) -> None:
  """Execute `clip_grad_norm_` to support clip grads in pytorch tests.

  Args:
      model: Neural network model instance under test.
      max_norm: Input `max_norm` parameter value (`float`).
  """
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
