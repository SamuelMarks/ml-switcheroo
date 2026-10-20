"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int, eps: float = 1e-05):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
        eps: Test argument or fixture.
    """
    self.eps = eps
    self.gamma = np.ones(normalized_shape)
    self.beta = np.zeros(normalized_shape)

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + self.eps)
    return self.gamma * x_norm + self.beta
