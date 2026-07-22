"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
import jax


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Helper to gelu activation."""
  return jax.nn.gelu(x, approximate=approximate)
