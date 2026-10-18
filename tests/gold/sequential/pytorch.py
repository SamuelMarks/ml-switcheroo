"""Test suite for the Pytorch module."""

import torch.nn as nn


def create_sequential(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))
