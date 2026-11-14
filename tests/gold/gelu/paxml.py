"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Execute `instantiate` to support gelu activation in PaxML tests.

  Args:
      x: Primary input tensor or array operand.
      approximate: Input `approximate` parameter value (`bool`).

  Returns:
      Resulting `jnp.ndarray` from gelu activation.
  """
  act = activations.GELU.HParams(approximate=approximate).instantiate()
  return act(x)
