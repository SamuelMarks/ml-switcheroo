"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class SameConvModel(nnx.Module):
  """Test suite for the Same Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, rngs: nnx.Rngs = None):
    """Initializes the SameConvModel instance."""
    self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding="SAME", rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.conv(x)
