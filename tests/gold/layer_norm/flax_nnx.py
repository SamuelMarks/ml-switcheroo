"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LayerNormModel(nnx.Module):
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int, rngs: nnx.Rngs):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.ln = nnx.LayerNorm(normalized_shape, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
