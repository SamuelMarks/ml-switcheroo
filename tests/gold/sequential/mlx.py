"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


def create_sequential(in_features: typing.Any, hidden: typing.Any, out_features: typing.Any) -> typing.Any:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.Sequential(nn.Linear(in_features, hidden), nn.relu, nn.Linear(hidden, out_features))
