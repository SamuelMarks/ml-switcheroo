"""Test suite for the Flax Nnx module."""

import typing
import jax.numpy as jnp

from flax import nnx


class SameConvModel(nnx.Module):  # type: ignore[misc]
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, rngs: typing.Optional[nnx.Rngs] = None):
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding="SAME", rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
