"""Module docstring."""

import numpy as np


def setup_adam(model, lr: float = 0.001):
  """Function docstring."""

  # <SWITCHEROO_FAILED_TO_TRANS>
  class NumpyAdam:
    """Class docstring."""

    def __init__(self, lr):
      """Function docstring."""
      self.lr = lr

  return NumpyAdam(lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
