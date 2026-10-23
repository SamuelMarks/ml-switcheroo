"""Test suite for the Paxml module."""

import jax.numpy as jnp

from praxis import optimizers


def setup_adam(model: jnp.ndarray, lr: float = 0.001) -> jnp.ndarray:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `jnp.ndarray` from setup adam.
  """
  return optimizers.Adam.HParams(learning_rate=lr)
