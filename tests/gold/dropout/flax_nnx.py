"""Test suite for the Flax Nnx module."""

import typing
import jax.numpy as jnp

from flax import nnx


class DropoutModel(nnx.Module):  # type: ignore[misc]
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5, rngs: typing.Optional[nnx.Rngs] = None):
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.dropout = nnx.Dropout(p, rngs=rngs)

  def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.
        deterministic: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x, deterministic=deterministic)
