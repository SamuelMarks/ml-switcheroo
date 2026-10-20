"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
import optax


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
