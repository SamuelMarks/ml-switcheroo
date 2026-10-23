"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
    """
    self.p = p

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    if training:
      mask = np.random.binomial(1, 1 - self.p, size=x.shape)
      return x * mask / (1 - self.p)
    return x
