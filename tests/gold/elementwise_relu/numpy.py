"""Test suite for the Numpy module."""

import typing

import numpy as np


def relu_activation(x: np.ndarray[typing.Any, typing.Any]) -> np.ndarray[typing.Any, typing.Any]:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return np.maximum(x, 0)
