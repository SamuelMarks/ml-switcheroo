"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax


def create_sequential(in_features: int, hidden: int, out_features: int, rngs: nnx.Rngs) -> nnx.Sequential:
  """Creates sequential."""
  return nnx.Sequential(
    nnx.Linear(in_features, hidden, rngs=rngs), jax.nn.relu, nnx.Linear(hidden, out_features, rngs=rngs)
  )
