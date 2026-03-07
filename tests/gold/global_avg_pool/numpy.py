"""Module docstring."""

import numpy as np


class GAPModel:
  """Class docstring."""

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    # Assuming NHWC format
    return np.mean(x, axis=(1, 2))
