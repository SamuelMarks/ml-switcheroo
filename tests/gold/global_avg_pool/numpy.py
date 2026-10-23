"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return np.mean(x, axis=(1, 2))
