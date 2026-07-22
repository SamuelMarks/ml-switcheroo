"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import poolings
import jax.numpy as jnp


class MaxPoolModel(base_layer.BaseLayer):
  """Test suite for the Max Pool Model component."""

  kernel_size: int = 2
  stride: int = 2

  def setup(self):
    """Helper to setup."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.pool(x)
