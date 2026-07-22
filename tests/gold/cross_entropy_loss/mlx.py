"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
  """Computes loss."""
  return nn.losses.cross_entropy(logits, targets, reduction="mean")
