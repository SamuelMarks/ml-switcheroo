"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class ConvModel(nnx.Module):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, rngs: nnx.Rngs = None):
    """Function docstring."""
    # Flax uses NHWC by default usually, but Conv can handle it
    self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.conv(x)
