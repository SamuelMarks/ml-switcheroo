"""Module docstring."""

import numpy as np


def create_sequential(in_features: int, hidden: int, out_features: int):
  """Function docstring."""

  # <SWITCHEROO_FAILED_TO_TRANS>
  # NumPy has no built-in sequential container
  class SequentialImpl:
    """Class docstring."""

    def __init__(self, in_feat, hid, out_feat):
      """Function docstring."""
      self.w1 = np.random.randn(in_feat, hid)
      self.b1 = np.zeros(hid)
      self.w2 = np.random.randn(hid, out_feat)
      self.b2 = np.zeros(out_feat)

    def __call__(self, x):
      """Function docstring."""
      x = np.dot(x, self.w1) + self.b1
      x = np.maximum(x, 0)
      return np.dot(x, self.w2) + self.b2

  return SequentialImpl(in_features, hidden, out_features)
  # </SWITCHEROO_FAILED_TO_TRANS>
