"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Helper to relu activation."""
  act = activations.ReLU.HParams().instantiate()
  return act(x)
