"""Test suite for the Numpy module."""

import typing

import numpy as np
from scipy.special import erf


def gelu_activation(x: np.ndarray[typing.Any, typing.Any], approximate: bool = False) -> typing.Any:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  if approximate:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
  return 0.5 * x * (1 + erf(x / np.sqrt(2)))
