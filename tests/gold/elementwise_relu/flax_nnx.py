"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
import jax


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Helper to relu activation."""
  return jax.nn.relu(x)
