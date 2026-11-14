"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


class LSTMModel(nnx.Module):
  """Minimal `nnx.Module` test double used in flax NNX tests."""

  def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Input `input_size` parameter value (`int`).
        hidden_size: Input `hidden_size` parameter value (`int`).
        rngs: Flax NNX Rngs container for stochastic parameter initialization.
    """

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
