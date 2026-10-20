"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def create_sequential(in_features: int, hidden: mx.array, out_features: int) -> mx.array:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.Sequential(nn.Linear(in_features, hidden), nn.relu, nn.Linear(hidden, out_features))
