"""Test suite for the Numpy module."""

import typing

import numpy as np


def bmm_einsum(x: np.ndarray[typing.Any, typing.Any], y: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.einsum("bik,bkj->bij", x, y)
