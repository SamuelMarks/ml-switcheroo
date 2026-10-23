"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Computes loss.

  Args:
      logits: The logits (torch.Tensor) for compute loss.
      targets: The targets (torch.Tensor) for compute loss.

  Returns:
      Resulting `torch.Tensor` from compute loss.
  """
  criterion = nn.CrossEntropyLoss()
  return criterion(logits, targets)
