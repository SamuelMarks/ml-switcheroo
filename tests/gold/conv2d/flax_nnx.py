"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

from flax import nnx


class ConvModel(nnx.Module):
  """Test helper class ConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, rngs: nnx.Rngs | None = None):
    """Initializes the ConvModel instance.

    Args:
        in_channels: Number of input convolution channels.
        out_channels: Number of output convolution channels.
        kernel_size: Convolution or pooling spatial kernel size.
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.conv(x)
