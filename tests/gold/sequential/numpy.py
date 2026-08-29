"""Test suite for the Numpy module."""

import typing

import numpy as np


def create_sequential(in_features: int, hidden: int, out_features: int) -> typing.Any:
  """Creates sequential."""

  class SequentialImpl:
    def __init__(self, in_feat: int, hid: int, out_feat: int) -> None:
      """Initializes the SequentialImpl instance."""
      self.w1 = np.random.randn(in_feat, hid)
      self.b1 = np.zeros(hid)
      self.w2 = np.random.randn(hid, out_feat)
      self.b2 = np.zeros(out_feat)

    def __call__(self, x: np.ndarray) -> np.ndarray:
      """Executes the callable instance."""
      x = np.dot(x, self.w1) + self.b1  # type: ignore
      x = np.maximum(x, 0)  # type: ignore
      return np.dot(x, self.w2) + self.b2  # type: ignore

  return SequentialImpl(in_features, hidden, out_features)
