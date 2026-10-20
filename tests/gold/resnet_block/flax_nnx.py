"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp
from flax import nnx


class ResidualBlock(nnx.Module):  # type: ignore[misc]
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int, rngs: nnx.Rngs):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
    self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        use_running_average: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
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
