"""Test suite for the Numpy module."""

import typing

import numpy as np


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    self.p = p

  def __call__(self, x: np.ndarray[typing.Any, typing.Any], training: bool = True) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if training:
      mask = np.random.binomial(1, 1 - self.p, size=x.shape)
      return x * mask / (1 - self.p)
    return x
