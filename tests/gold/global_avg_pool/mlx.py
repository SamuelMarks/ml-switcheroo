"""Test suite for the Mlx module."""

import typing

import mlx.core as mx
import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return mx.mean(x, axis=(1, 2))
