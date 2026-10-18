"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn
import mlx.optimizers as optim

nn: typing.Any = _mlx_nn


def setup_adam(model: typing.Any, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optim.Adam(learning_rate=lr)
