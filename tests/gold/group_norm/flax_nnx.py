"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class GNModel(nnx.Module):
  """Test suite for the G N Model component."""

  def __init__(self, num_groups: int, num_channels: int, rngs: nnx.Rngs):
    """Initializes the GNModel instance."""
    self.gn = nnx.GroupNorm(num_groups=num_groups, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.gn(x)
