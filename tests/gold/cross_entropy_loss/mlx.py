"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.losses.cross_entropy(logits, targets, reduction="mean")
