"""Test suite for the Paxml module."""

import typing
from praxis import base_layer  # type: ignore
from praxis.layers import poolings  # type: ignore


class MaxPoolModel(base_layer.BaseLayer):  # type: ignore
  """Test suite for the Max Pool Model component."""

  kernel_size: int = 2
  stride: int = 2

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    return self.pool(x)
