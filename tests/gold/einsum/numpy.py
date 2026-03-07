"""Module docstring."""

import numpy as np


def bmm_einsum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Function docstring."""
  # NumPy
  return np.einsum("bik,bkj->bij", x, y)
