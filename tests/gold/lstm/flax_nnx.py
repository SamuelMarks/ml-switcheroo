"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LSTMModel(nnx.Module):
  """Docstring."""

  def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs):
    """Initializes the LSTMModel instance."""
    pass

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    pass
