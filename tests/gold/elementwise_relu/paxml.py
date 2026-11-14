"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis.layers import activations


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Execute `instantiate` to support relu activation in PaxML tests.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `jnp.ndarray` from relu activation.
  """
  act = activations.ReLU.HParams().instantiate()
  return act(x)
