"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class MaxPoolModel(nnx.Module):
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: jnp.ndarray = 2, stride: jnp.ndarray = 2, rngs: nnx.Rngs | None = None):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Convolution or pooling spatial kernel size.
        stride: Convolution or pooling stride.
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.kernel_size = kernel_size
    self.stride = stride

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    import jax.lax as lax

    return lax.reduce_window(
      x, -jnp.inf, lax.max, (1, self.kernel_size, self.kernel_size, 1), (1, self.stride, self.stride, 1), "VALID"
    )
