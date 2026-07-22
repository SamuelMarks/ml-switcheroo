"""Test suite for the Numpy module."""

import numpy as np


class DropoutModel:
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance."""
    self.p = p

  def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
    """Executes the callable instance."""
    if training:
      mask = np.random.binomial(1, 1 - self.p, size=x.shape)
      return x * mask / (1 - self.p)
    return x
