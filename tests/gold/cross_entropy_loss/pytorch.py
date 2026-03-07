"""Module docstring."""

import torch
import torch.nn as nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Function docstring."""
  # PyTorch combines LogSoftmax and NLLLoss in CrossEntropyLoss
  # Targets are expected to be class indices
  criterion = nn.CrossEntropyLoss()
  return criterion(logits, targets)
