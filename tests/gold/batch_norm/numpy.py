"""Test suite for the Numpy module."""

import typing

import numpy as np


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int, momentum: float = 0.1):
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
        momentum: Test argument or fixture.
    """
    self.gamma = np.ones(num_features)
    self.beta = np.zeros(num_features)
    self.running_mean = np.zeros(num_features)
    self.running_var = np.ones(num_features)
    self.momentum = momentum

  def __call__(self, x: np.ndarray[typing.Any, typing.Any], training: bool = True) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if training:
      mean = np.mean(x, axis=(0, 1, 2))
      var = np.var(x, axis=(0, 1, 2))
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var
    x_norm = (x - mean) / np.sqrt(var + 1e-05)
    return self.gamma * x_norm + self.beta
