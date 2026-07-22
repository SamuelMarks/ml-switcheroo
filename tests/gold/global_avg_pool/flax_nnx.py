"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class GAPModel(nnx.Module):
  """Test suite for the G A P Model component."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return jnp.mean(x, axis=(1, 2))
