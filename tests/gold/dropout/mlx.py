"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: typing.Any = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x)
