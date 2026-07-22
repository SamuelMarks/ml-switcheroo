"""Test suite for the Numpy module."""

import numpy as np


def relu_activation(x: np.ndarray) -> np.ndarray:
  """Helper to relu activation."""
  return np.maximum(x, 0)
