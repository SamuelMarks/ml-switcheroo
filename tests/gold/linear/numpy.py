"""Test suite for the Numpy module."""

import typing

import numpy as np


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    self.weight = np.random.randn(in_features, out_features) / np.sqrt(in_features)
    self.bias = np.zeros(out_features)

  def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return np.dot(x, self.weight) + self.bias
