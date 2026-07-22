"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Helper to gelu activation."""
  act = activations.GELU.HParams(approximate=approximate).instantiate()
  return act(x)
