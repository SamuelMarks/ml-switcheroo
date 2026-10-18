"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jax.nn.gelu(x, approximate=approximate)
