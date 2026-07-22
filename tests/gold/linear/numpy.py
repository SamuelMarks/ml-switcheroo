"""Test suite for the Numpy module."""

import numpy as np


class Model:
  """Test suite for the Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance."""
    self.weight = np.random.randn(in_features, out_features) / np.sqrt(in_features)
    self.bias = np.zeros(out_features)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    return np.dot(x, self.weight) + self.bias
