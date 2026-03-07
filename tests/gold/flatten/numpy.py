"""Module docstring."""

import numpy as np


class FlattenModel:
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    self.start_dim = start_dim

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
