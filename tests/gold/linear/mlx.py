"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: typing.Any, out_features: typing.Any):
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
