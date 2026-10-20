"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    self.p = p

  def __call__(self, x: np.typing.NDArray[np.float64], training: bool = True) -> np.typing.NDArray[np.float64]:
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
