"""Test suite for the Numpy module."""

import numpy as np


class LayerNormModel:
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int, eps: float = 1e-05):
    """Initializes the LayerNormModel instance."""
    self.eps = eps
    self.gamma = np.ones(normalized_shape)
    self.beta = np.zeros(normalized_shape)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Executes the callable instance."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + self.eps)
    return self.gamma * x_norm + self.beta
