"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class LSTMModel:
  """Test helper class LSTMModel."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: The input size (int) for   init  .
        hidden_size: The hidden size (int) for   init  .
    """
    pass

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
