"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

from flax import nnx


class GNModel(nnx.Module):
  """Test helper class GNModel."""

  def __init__(self, num_groups: int, num_channels: int, rngs: jnp.ndarray):
    """Initializes the GNModel instance.

    Args:
        num_groups: The num groups (int) for   init  .
        num_channels: The num channels (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.gn = getattr(nnx, "GroupNorm")(num_groups=num_groups, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.gn(x)
