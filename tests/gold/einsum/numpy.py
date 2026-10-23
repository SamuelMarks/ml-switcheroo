"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def bmm_einsum(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
  """Helper to bmm einsum.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `np.typing.NDArray[np.float64]` from bmm einsum.
  """
  return np.asarray(np.einsum("bik,bkj->bij", x, y), dtype=np.float64)
