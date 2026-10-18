"""Test suite for the Flax Nnx module."""

import jax
from flax import nnx


def create_sequential(in_features: int, hidden: int, out_features: int, rngs: nnx.Rngs) -> nnx.Sequential:
  """Creates sequential.

  Args:
      in_features: Test argument or fixture.
      hidden: Test argument or fixture.
      out_features: Test argument or fixture.
      rngs: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nnx.Sequential(
    nnx.Linear(in_features, hidden, rngs=rngs), jax.nn.relu, nnx.Linear(hidden, out_features, rngs=rngs)
  )
