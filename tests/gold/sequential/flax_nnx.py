"""Test suite for the Flax Nnx module."""

import jax
from flax import nnx


def create_sequential(in_features: int, hidden: int, out_features: int, rngs: nnx.Rngs) -> nnx.Sequential:
  """Creates sequential.

  Args:
      in_features: Number of input linear features.
      hidden: The hidden (int) for create sequential.
      out_features: Number of output linear features.
      rngs: Flax NNX Rngs container for stochastic parameter initialization.

  Returns:
      Resulting `nnx.Sequential` from create sequential.
  """
  return nnx.Sequential(
    nnx.Linear(in_features, hidden, rngs=rngs), jax.nn.relu, nnx.Linear(hidden, out_features, rngs=rngs)
  )
