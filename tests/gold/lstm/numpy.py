"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class LSTMModel:
  """Sentinel `LSTMModel` fixture class used in numpy tests."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Input `input_size` parameter value (`int`).
        hidden_size: Input `hidden_size` parameter value (`int`).
    """

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
