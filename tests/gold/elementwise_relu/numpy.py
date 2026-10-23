"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def relu_activation(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
  """Helper to relu activation.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `np.typing.NDArray[np.float64]` from relu activation.
  """
  return np.maximum(x, 0)
