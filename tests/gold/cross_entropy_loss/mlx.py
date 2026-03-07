"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
  """Function docstring."""
  # MLX has cross_entropy
  return nn.losses.cross_entropy(logits, targets, reduction="mean")
