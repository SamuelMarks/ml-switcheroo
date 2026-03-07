"""Module docstring."""

from flax import nnx
import jax.numpy as jnp


class Model(nnx.Module):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    """Function docstring."""
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    return self.linear(x)
