"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Execute `relu` to support relu activation in flax NNX tests.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `jnp.ndarray` from relu activation.
  """
  return jax.nn.relu(x)
