"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing
from scipy.special import erf


def gelu_activation(x: np.typing.NDArray[np.float64], approximate: bool = False) -> np.typing.NDArray[np.float64]:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  if approximate:
    return np.asarray(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))), dtype=np.float64)
  return np.asarray(0.5 * x * (1 + erf(x / np.sqrt(2))), dtype=np.float64)
