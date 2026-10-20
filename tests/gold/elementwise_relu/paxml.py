"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  act = activations.ReLU.HParams().instantiate()
  return act(x)
