"""Test suite for the Numpy module."""

import numpy as np


class GAPModel:
  """Test suite for the G A P Model component."""

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    return np.mean(x, axis=(1, 2))
