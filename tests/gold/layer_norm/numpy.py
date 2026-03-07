"""Module docstring."""

import numpy as np


class LayerNormModel:
  """Class docstring."""

  def __init__(self, normalized_shape: int, eps: float = 1e-5):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.eps = eps
    self.gamma = np.ones(normalized_shape)
    self.beta = np.zeros(normalized_shape)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + self.eps)
    return self.gamma * x_norm + self.beta
    # </SWITCHEROO_FAILED_TO_TRANS>
