"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


class DropoutModel(nnx.Module):
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5, rngs: nnx.Rngs = None):
    """Initializes the DropoutModel instance."""
    self.dropout = nnx.Dropout(p, rngs=rngs)

  def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    """Executes the callable instance."""
    return self.dropout(x, deterministic=deterministic)
