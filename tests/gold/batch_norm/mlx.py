"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: typing.Any):
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
    """
    super().__init__()
    self.bn = nn.BatchNorm(num_features)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x)
