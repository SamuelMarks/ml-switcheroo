"""Module docstring."""

from praxis import base_layer
from praxis.layers import poolings
import jax.numpy as jnp


class MaxPoolModel(base_layer.BaseLayer):
  """Class docstring."""

  kernel_size: int = 2
  stride: int = 2

  def setup(self):
    """Function docstring."""
    self.create_child(
      "pool",
      poolings.Pooling.HParams(
        window_shape=(self.kernel_size, self.kernel_size), window_stride=(self.stride, self.stride), pooling_type="MAX"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.pool(x)
