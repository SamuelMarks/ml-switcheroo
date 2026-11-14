"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class ResidualBlock:
  """Sentinel `ResidualBlock` fixture class used in numpy tests."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Input `channels` parameter value (`int`).
    """

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool`).

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
