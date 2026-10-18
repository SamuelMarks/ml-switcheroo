"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: typing.Any = 2, stride: typing.Any = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Test argument or fixture.
        stride: Test argument or fixture.
    """
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.pool(x)
