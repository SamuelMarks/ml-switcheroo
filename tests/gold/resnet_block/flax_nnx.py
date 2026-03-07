"""Module docstring."""

from flax import nnx
import jax.numpy as jnp
import jax


class ResidualBlock(nnx.Module):
  """Class docstring."""

  def __init__(self, channels: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
    self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Function docstring."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, use_running_average=use_running_average)
    out = jax.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, use_running_average=use_running_average)
    out = out + residual
    out = jax.nn.relu(out)
    return out
