"""Test suite for the Paxml module."""

import typing


class FlattenModel:
  """Test helper class FlattenModel."""

  start_dim: typing.Any = 1

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    batch_shape: typing.Any = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
