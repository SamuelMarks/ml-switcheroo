"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp

from flax import nnx


class DropoutModel(nnx.Module):
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5, rngs: nnx.Rngs | None = None):
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.dropout = nnx.Dropout(p, rngs=rngs)

  def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.
        deterministic: The deterministic (bool) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.dropout(x, deterministic=deterministic)
