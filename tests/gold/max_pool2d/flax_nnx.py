"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class MaxPoolModel(nnx.Module):
  """Test suite for the Max Pool Model component."""

  def __init__(self, kernel_size: int = 2, stride: int = 2, rngs: nnx.Rngs = None):
    """Initializes the MaxPoolModel instance."""
    self.kernel_size = kernel_size
    self.stride = stride

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    import jax.lax as lax

    return lax.reduce_window(
      x, -jnp.inf, lax.max, (1, self.kernel_size, self.kernel_size, 1), (1, self.stride, self.stride, 1), "VALID"
    )
