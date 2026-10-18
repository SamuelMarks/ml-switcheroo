"""Test suite for the Numpy module."""

import typing

import numpy as np


def concat_tensors(
  x: np.ndarray[typing.Any, typing.Any], y: np.ndarray[typing.Any, typing.Any], axis: int = -1
) -> np.ndarray[typing.Any, typing.Any]:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.concatenate([x, y], axis=axis)
