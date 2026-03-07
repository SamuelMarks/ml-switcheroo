"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class LayerNormModel(nnx.Module):
  """Class docstring."""

  def __init__(self, normalized_shape: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.ln = nnx.LayerNorm(normalized_shape, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.ln(x)
