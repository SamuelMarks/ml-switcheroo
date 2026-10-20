"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class SameConvModel:
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    pass

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x
