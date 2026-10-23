"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def create_sequential(in_features: int, hidden: mx.array, out_features: int) -> mx.array:
  """Creates sequential.

  Args:
      in_features: Number of input linear features.
      hidden: The hidden (mx.array) for create sequential.
      out_features: Number of output linear features.

  Returns:
      Resulting `mx.array` from create sequential.
  """
  return nn.Sequential(nn.Linear(in_features, hidden), nn.relu, nn.Linear(hidden, out_features))
