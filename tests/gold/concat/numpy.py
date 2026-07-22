"""Test suite for the Numpy module."""

import numpy as np


def concat_tensors(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
  """Helper to concat tensors."""
  return np.concatenate([x, y], axis=axis)
