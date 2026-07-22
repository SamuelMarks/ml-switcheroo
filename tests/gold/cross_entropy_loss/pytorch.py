"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Computes loss."""
  criterion = nn.CrossEntropyLoss()
  return criterion(logits, targets)
