"""Module docstring."""

import numpy as np
from scipy.special import erf


def gelu_activation(x: np.ndarray, approximate: bool = False) -> np.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  if approximate:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
  return 0.5 * x * (1 + erf(x / np.sqrt(2)))
  # </SWITCHEROO_FAILED_TO_TRANS>
