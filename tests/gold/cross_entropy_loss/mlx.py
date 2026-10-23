"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
  """Computes loss.

  Args:
      logits: The logits (mx.array) for compute loss.
      targets: The targets (mx.array) for compute loss.

  Returns:
      Resulting `mx.array` from compute loss.
  """
  return nn.losses.cross_entropy(logits, targets, reduction="mean")
