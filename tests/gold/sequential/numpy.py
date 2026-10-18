"""Test suite for the Numpy module."""

import typing

import numpy as np


def create_sequential(in_features: int, hidden: int, out_features: int) -> typing.Any:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  class SequentialImpl:
    """Test helper class SequentialImpl."""

    def __init__(self, in_feat: int, hid: int, out_feat: int) -> None:
      """Initializes the SequentialImpl instance.

      Args:
          in_feat: Test argument or fixture.
          hid: Test argument or fixture.
          out_feat: Test argument or fixture.
      """
      self.w1 = np.random.randn(in_feat, hid)
      self.b1 = np.zeros(hid)
      self.w2 = np.random.randn(hid, out_feat)
      self.b2 = np.zeros(out_feat)

    def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
      """Executes the callable instance.

      Args:
          x: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      x = np.dot(x, self.w1) + self.b1
      x = np.maximum(x, 0)
      return np.dot(x, self.w2) + self.b2

  return SequentialImpl(in_features, hidden, out_features)
