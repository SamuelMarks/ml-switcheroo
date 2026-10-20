"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  criterion = nn.CrossEntropyLoss()
  return criterion(logits, targets)
