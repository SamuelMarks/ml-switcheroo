"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jax.nn.relu(x)
