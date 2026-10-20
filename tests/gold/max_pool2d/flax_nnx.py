"""Test suite for the Flax Nnx module."""

import typing

import jax.numpy as jnp
from flax import nnx


class MaxPoolModel(nnx.Module):  # type: ignore[misc]
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: jnp.ndarray = 2, stride: jnp.ndarray = 2, rngs: typing.Optional[nnx.Rngs] = None):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Test argument or fixture.
        stride: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.kernel_size = kernel_size
    self.stride = stride

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    import jax.lax as lax

    return lax.reduce_window(
      x, -jnp.inf, lax.max, (1, self.kernel_size, self.kernel_size, 1), (1, self.stride, self.stride, 1), "VALID"
    )
