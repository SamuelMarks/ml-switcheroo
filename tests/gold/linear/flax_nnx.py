"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class Model(nnx.Module):  # type: ignore[misc]
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
