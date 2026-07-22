"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def clip_grads(model: nn.Module, max_norm: float = 1.0):
  """Helper to clip grads."""
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
