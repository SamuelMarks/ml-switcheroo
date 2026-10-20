"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    pass

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x
