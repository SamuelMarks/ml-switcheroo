"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def relu_activation(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.maximum(x, 0)
