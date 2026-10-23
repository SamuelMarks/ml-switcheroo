"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    self.weight = np.random.randn(in_features, out_features) / np.sqrt(in_features)
    self.bias = np.zeros(out_features)

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return np.asarray(np.dot(x, self.weight) + self.bias, dtype=np.float64)
