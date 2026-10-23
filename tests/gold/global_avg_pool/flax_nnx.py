"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class GAPModel(nnx.Module):
  """Test helper class GAPModel."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return jnp.mean(x, axis=(1, 2))
