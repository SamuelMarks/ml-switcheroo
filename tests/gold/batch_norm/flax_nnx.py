"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class BNModel(nnx.Module):
  """Test helper class BNModel."""

  def __init__(self, num_features: int, rngs: nnx.Rngs):
    """Initializes the BNModel instance.

    Args:
        num_features: The num features (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.bn = nnx.BatchNorm(num_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        use_running_average: The use running average (bool) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.bn(x, use_running_average=use_running_average)
