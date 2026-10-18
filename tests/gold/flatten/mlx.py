"""Test suite for the Mlx module."""

import typing

import mlx.core as mx
import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: typing.Any = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Test argument or fixture.
    """
    super().__init__()
    self.start_dim = start_dim

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return mx.flatten(x, start_axis=self.start_dim)
