"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Helper to relu activation."""
  return jax.nn.relu(x)
