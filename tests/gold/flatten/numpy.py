"""Test suite for the Numpy module."""

import numpy as np


class FlattenModel:
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance."""
    self.start_dim = start_dim

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
