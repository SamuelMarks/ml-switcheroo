"""Module docstring."""

import numpy as np


class MLP:
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.w1 = np.random.randn(in_features, hidden_features) / np.sqrt(in_features)
    self.b1 = np.zeros(hidden_features)
    self.w2 = np.random.randn(hidden_features, out_features) / np.sqrt(hidden_features)
    self.b2 = np.zeros(out_features)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    x = np.dot(x, self.w1) + self.b1
    x = np.maximum(x, 0)
    x = np.dot(x, self.w2) + self.b2
    return x
