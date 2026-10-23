"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp
from flax import nnx


class ResidualBlock(nnx.Module):
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int, rngs: nnx.Rngs):
    """Initializes the ResidualBlock instance.

    Args:
        channels: The channels (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
    self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        use_running_average: The use running average (bool) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, use_running_average=use_running_average)
    out = jax.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, use_running_average=use_running_average)
    out = out + residual
    out = jax.nn.relu(out)
    return out
