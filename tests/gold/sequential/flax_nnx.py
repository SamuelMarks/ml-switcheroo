"""Module docstring."""

from flax import nnx
import jax.numpy as jnp
import jax


def create_sequential(in_features: int, hidden: int, out_features: int, rngs: nnx.Rngs) -> nnx.Sequential:
  """Function docstring."""
  return nnx.Sequential(
    nnx.Linear(in_features, hidden, rngs=rngs), jax.nn.relu, nnx.Linear(hidden, out_features, rngs=rngs)
  )
