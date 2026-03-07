"""Module docstring."""

import jax.numpy as jnp
import jax


def relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  return jax.nn.relu(x)
