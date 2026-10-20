"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Test argument or fixture.
    """
    self.start_dim = start_dim

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
