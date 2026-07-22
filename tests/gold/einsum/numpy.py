"""Test suite for the Numpy module."""

import numpy as np


def bmm_einsum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Helper to bmm einsum."""
  return np.einsum("bik,bkj->bij", x, y)
