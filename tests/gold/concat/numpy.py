"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def concat_tensors(
  x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], axis: int = -1
) -> np.typing.NDArray[np.float64]:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: The axis (int) for concat tensors.

  Returns:
      Resulting `np.typing.NDArray[np.float64]` from concat tensors.
  """
  return np.concatenate([x, y], axis=axis)
