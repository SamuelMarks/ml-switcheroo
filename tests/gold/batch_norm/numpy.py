"""Module docstring."""

import numpy as np


class BNModel:
  """Class docstring."""

  def __init__(self, num_features: int, momentum: float = 0.1):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.gamma = np.ones(num_features)
    self.beta = np.zeros(num_features)
    self.running_mean = np.zeros(num_features)
    self.running_var = np.ones(num_features)
    self.momentum = momentum
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    if training:
      mean = np.mean(x, axis=(0, 1, 2))
      var = np.var(x, axis=(0, 1, 2))
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    return self.gamma * x_norm + self.beta
    # </SWITCHEROO_FAILED_TO_TRANS>
