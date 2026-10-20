"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def concat_tensors(
  x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], axis: int = -1
) -> np.typing.NDArray[np.float64]:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.concatenate([x, y], axis=axis)
