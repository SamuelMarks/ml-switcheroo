"""Test suite for the Numpy module."""

import numpy as np


class MLP:
  """Test suite for the M L P component."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance."""
    self.w1 = np.random.randn(in_features, hidden_features) / np.sqrt(in_features)
    self.b1 = np.zeros(hidden_features)
    self.w2 = np.random.randn(hidden_features, out_features) / np.sqrt(hidden_features)
    self.b2 = np.zeros(out_features)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    x = np.dot(x, self.w1) + self.b1
    x = np.maximum(x, 0)
    x = np.dot(x, self.w2) + self.b2
    return x
