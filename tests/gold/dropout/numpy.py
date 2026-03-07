"""Module docstring."""

import numpy as np


class DropoutModel:
  """Class docstring."""

  def __init__(self, p: float = 0.5):
    """Function docstring."""
    self.p = p

  def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    if training:
      mask = np.random.binomial(1, 1 - self.p, size=x.shape)
      return x * mask / (1 - self.p)
    return x
    # </SWITCHEROO_FAILED_TO_TRANS>
