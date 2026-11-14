"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
  """Computes loss.

  Args:
      logits: Input `logits` parameter value (`mx.array`).
      targets: Input `targets` parameter value (`mx.array`).

  Returns:
      Resulting `mx.array` from compute loss.
  """
  return nn.losses.cross_entropy(logits, targets, reduction="mean")
