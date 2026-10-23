"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import poolings


class MaxPoolModel(base_layer.BaseLayer):
  """Test helper class MaxPoolModel."""

  kernel_size = 2
  stride = 2

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.pool(x)
