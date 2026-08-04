"""Example of basic mathematical operations built using JAX.

This module defines a simple Mean Absolute Error loss calculation function,
demonstrating the usage of JAX NumPy APIs and providing a translation target
or source for framework mapping.
"""

import jax.numpy as jnp


def compute_loss(prediction, target):
  """Calculates Mean Absolute Error.

  Semantic pivot: jnp.abs, jnp.mean -> torch.abs, torch.mean.

  Args:
    prediction (jax.Array): The predicted values as a JAX array.
    target (jax.Array): The target/ground-truth values as a JAX array.

  Returns:
    jax.Array: The calculated Mean Absolute Error loss as a scalar JAX array.
  """
  diff = jnp.abs(prediction - target)
  loss = jnp.mean(diff)
  return loss
