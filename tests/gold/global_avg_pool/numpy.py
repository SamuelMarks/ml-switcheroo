"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return np.mean(x, axis=(1, 2))
