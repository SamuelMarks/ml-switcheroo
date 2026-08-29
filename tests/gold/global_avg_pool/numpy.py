"""Test suite for the Numpy module."""

import numpy as np


class GAPModel:
  """Docstring."""

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    return np.mean(x, axis=(1, 2))
