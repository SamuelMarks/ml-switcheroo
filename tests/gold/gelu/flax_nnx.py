"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (bool) for gelu activation.

  Returns:
      Resulting `jnp.ndarray` from gelu activation.
  """
  return jax.nn.gelu(x, approximate=approximate)
