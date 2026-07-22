"""Test suite for the Numpy module."""

import numpy as np
from scipy.special import erf


def gelu_activation(x: np.ndarray, approximate: bool = False) -> np.ndarray:
  """Helper to gelu activation."""
  if approximate:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
  return 0.5 * x * (1 + erf(x / np.sqrt(2)))
