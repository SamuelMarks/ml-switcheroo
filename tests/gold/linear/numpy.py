"""Module docstring."""

import numpy as np


class Model:
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Manual parameter initialization
    self.weight = np.random.randn(in_features, out_features) / np.sqrt(in_features)
    self.bias = np.zeros(out_features)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    return np.dot(x, self.weight) + self.bias
