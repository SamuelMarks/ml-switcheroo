"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def bmm_einsum(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.asarray(np.einsum("bik,bkj->bij", x, y), dtype=np.float64)
