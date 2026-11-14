"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as _mlx_nn
import mlx.optimizers as optim

nn: mx.array = _mlx_nn


def setup_adam(model: mx.array, lr: float = 0.001) -> mx.array:
  """Execute `optim.Adam` to support setup adam in MLX tests.

  Args:
      model: Neural network model instance under test.
      lr: Input `lr` parameter value (`float`).

  Returns:
      Resulting `mx.array` from setup adam.
  """
  return optim.Adam(learning_rate=lr)
