"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: typing.Any):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
