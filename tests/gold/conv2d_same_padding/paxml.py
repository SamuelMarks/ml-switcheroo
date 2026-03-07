"""Module docstring."""

from praxis import base_layer
from praxis.layers import convolutions
import jax.numpy as jnp


class SameConvModel(base_layer.BaseLayer):
  """Class docstring."""

  in_channels: int = 0
  out_channels: int = 0
  kernel_size: int = 3

  def setup(self):
    """Function docstring."""
    self.create_child(
      "conv",
      convolutions.Conv2D.HParams(
        filter_shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels), padding="SAME"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.conv(x)
