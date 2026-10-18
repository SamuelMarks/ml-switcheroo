"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


def relu_activation(x: typing.Any) -> typing.Any:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.relu(x)
