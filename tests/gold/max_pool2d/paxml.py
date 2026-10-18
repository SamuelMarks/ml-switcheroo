"""Test suite for the Paxml module."""

import typing

from praxis.layers import poolings


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  kernel_size: typing.Any = 2
  stride: typing.Any = 2

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.pool(x)
