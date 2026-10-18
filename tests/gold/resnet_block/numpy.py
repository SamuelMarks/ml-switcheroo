"""Test suite for the Numpy module."""

import typing

import numpy as np


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    pass

  def __call__(self, x: np.ndarray[typing.Any, typing.Any], training: bool = True) -> np.ndarray[typing.Any, typing.Any]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x
