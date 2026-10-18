"""Test suite for the Numpy module."""

import typing

import numpy as np


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

  def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
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
