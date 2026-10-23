"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: The channels (int) for   init  .
    """
    pass

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
