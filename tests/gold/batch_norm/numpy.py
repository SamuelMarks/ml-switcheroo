"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class BNModel:
  """Sentinel `BNModel` fixture class used in numpy tests."""

  def __init__(self, num_features: int, momentum: float = 0.1):
    """Initializes the BNModel instance.

    Args:
        num_features: Input `num_features` parameter value (`int`).
        momentum: Input `momentum` parameter value (`float`).
    """
    self.gamma = np.ones(num_features)
    self.beta = np.zeros(num_features)
    self.running_mean = np.zeros(num_features)
    self.running_var = np.ones(num_features)
    self.momentum = momentum

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool`).

    Returns:
        Output tensor or value computed by the forward pass.
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
