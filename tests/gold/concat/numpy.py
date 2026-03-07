"""Module docstring."""

import numpy as np


def concat_tensors(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
  """Function docstring."""
  return np.concatenate([x, y], axis=axis)
