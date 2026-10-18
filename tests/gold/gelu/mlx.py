"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


def gelu_activation(x: typing.Any, approximate: typing.Any = "none") -> typing.Any:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
