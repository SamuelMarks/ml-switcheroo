"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class BNModel(nnx.Module):
  """Test suite for the B N Model component."""

  def __init__(self, num_features: int, rngs: nnx.Rngs):
    """Initializes the BNModel instance."""
    self.bn = nnx.BatchNorm(num_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.bn(x, use_running_average=use_running_average)
