"""Test suite for the Numpy module."""

import numpy as np


def create_sequential(in_features: int, hidden: int, out_features: int):
  """Creates sequential."""

  class SequentialImpl:
    """Test suite for the Sequential Impl component."""

    def __init__(self, in_feat, hid, out_feat):
      """Initializes the SequentialImpl instance."""
      self.w1 = np.random.randn(in_feat, hid)
      self.b1 = np.zeros(hid)
      self.w2 = np.random.randn(hid, out_feat)
      self.b2 = np.zeros(out_feat)

    def __call__(self, x):
      """Executes the callable instance."""
      x = np.dot(x, self.w1) + self.b1
      x = np.maximum(x, 0)
      return np.dot(x, self.w2) + self.b2

  return SequentialImpl(in_features, hidden, out_features)
