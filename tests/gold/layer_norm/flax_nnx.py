"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LayerNormModel(nnx.Module):
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int, rngs: nnx.Rngs):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: The normalized shape (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    self.ln = nnx.LayerNorm(normalized_shape, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.ln(x)
