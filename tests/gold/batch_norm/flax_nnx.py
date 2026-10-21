"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class BNModel(nnx.Module):
  """Test helper class BNModel."""

  def __init__(self, num_features: int, rngs: nnx.Rngs):
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.bn = nnx.BatchNorm(num_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        use_running_average: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x, use_running_average=use_running_average)
