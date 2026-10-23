"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LSTMModel(nnx.Module):
  """Test helper class LSTMModel."""

  def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs):
    """Initializes the LSTMModel instance.

    Args:
        input_size: The input size (int) for   init  .
        hidden_size: The hidden size (int) for   init  .
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """
    pass

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
