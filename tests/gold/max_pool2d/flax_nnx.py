"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class MaxPoolModel(nnx.Module):
  """Class docstring."""

  def __init__(self, kernel_size: int = 2, stride: int = 2, rngs: nnx.Rngs = None):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # NNX might not have a stateful maxpool layer, but uses flax.linen.max_pool or similar functional approach usually.
    # Assuming functional approach here as nnx layers are still evolving for simple non-param ops
    self.kernel_size = kernel_size
    self.stride = stride
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    import jax.lax as lax

    return lax.reduce_window(
      x, -jnp.inf, lax.max, (1, self.kernel_size, self.kernel_size, 1), (1, self.stride, self.stride, 1), "VALID"
    )
    # </SWITCHEROO_FAILED_TO_TRANS>
