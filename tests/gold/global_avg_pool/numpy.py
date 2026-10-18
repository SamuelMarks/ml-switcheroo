"""Test suite for the Numpy module."""

import typing

import numpy as np


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return np.mean(x, axis=(1, 2))
