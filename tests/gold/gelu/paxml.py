"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (bool) for gelu activation.

  Returns:
      Resulting `jnp.ndarray` from gelu activation.
  """
  act = activations.GELU.HParams(approximate=approximate).instantiate()
  return act(x)
