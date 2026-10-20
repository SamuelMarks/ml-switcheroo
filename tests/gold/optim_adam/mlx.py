"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as _mlx_nn
import mlx.optimizers as optim

nn: mx.array = _mlx_nn


def setup_adam(model: mx.array, lr: float = 0.001) -> mx.array:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optim.Adam(learning_rate=lr)
