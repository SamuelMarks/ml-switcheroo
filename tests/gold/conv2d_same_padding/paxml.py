"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import convolutions


class SameConvModel(base_layer.BaseLayer):
  """Test helper class SameConvModel."""

  in_channels = 0
  out_channels = 0
  kernel_size = 3

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "conv",
      convolutions.Conv2D.HParams(
        filter_shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels), padding="SAME"
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.conv(x)
