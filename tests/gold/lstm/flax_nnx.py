"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LSTMModel(nnx.Module):  # type: ignore[misc]
  """Test helper class LSTMModel."""

  def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Test argument or fixture.
        hidden_size: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    pass

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x
