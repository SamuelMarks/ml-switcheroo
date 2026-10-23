"""Test suite for the Pytorch module."""

import torch.nn as nn


def create_sequential(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
  """Creates sequential.

  Args:
      in_features: Number of input linear features.
      hidden: The hidden (int) for create sequential.
      out_features: Number of output linear features.

  Returns:
      Resulting `nn.Sequential` from create sequential.
  """
  return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))
