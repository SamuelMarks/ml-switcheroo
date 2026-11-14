"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class MLP:
  """Sentinel `MLP` fixture class used in numpy tests."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: Number of hidden linear features.
        out_features: Number of output linear features.
    """
    self.w1: np.typing.NDArray[np.float64] = np.asarray(
      np.random.randn(in_features, hidden_features) / np.sqrt(in_features), dtype=np.float64
    )
    self.b1: np.typing.NDArray[np.float64] = np.zeros(hidden_features, dtype=np.float64)
    self.w2: np.typing.NDArray[np.float64] = np.asarray(
      np.random.randn(hidden_features, out_features) / np.sqrt(hidden_features), dtype=np.float64
    )
    self.b2: np.typing.NDArray[np.float64] = np.zeros(out_features, dtype=np.float64)

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    hidden: np.typing.NDArray[np.float64] = np.maximum(np.dot(x, self.w1) + self.b1, 0.0)
    return np.asarray(np.dot(hidden, self.w2) + self.b2, dtype=np.float64)
