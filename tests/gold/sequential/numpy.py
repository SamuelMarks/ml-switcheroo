"""Test suite for the Numpy module."""

import typing

import numpy as np
import numpy.typing


def create_sequential(
  in_features: int, hidden: int, out_features: int
) -> typing.Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
  """Creates sequential.

  Args:
      in_features: Number of input linear features.
      hidden: The hidden (int) for create sequential.
      out_features: Number of output linear features.

  Returns:
      Resulting `typing.Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]` from create sequential.
  """

  class SequentialImpl:
    """Test helper class SequentialImpl."""

    def __init__(self, in_feat: int, hid: int, out_feat: int) -> None:
      """Initializes the SequentialImpl instance.

      Args:
          in_feat: The in feat (int) for   init  .
          hid: The hid (int) for   init  .
          out_feat: The out feat (int) for   init  .
      """
      self.w1 = np.random.randn(in_feat, hid)
      self.b1 = np.zeros(hid)
      self.w2 = np.random.randn(hid, out_feat)
      self.b2 = np.zeros(out_feat)

    def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
      """Executes the callable instance.

      Args:
          x: Primary input tensor or array operand.

      Returns:
          Output tensor or value computed by the forward pass.
      """
      x = np.dot(x, self.w1) + self.b1
      x = np.maximum(x, 0)
      return np.asarray(np.dot(x, self.w2) + self.b2, dtype=np.float64)

  return SequentialImpl(in_features, hidden, out_features)
