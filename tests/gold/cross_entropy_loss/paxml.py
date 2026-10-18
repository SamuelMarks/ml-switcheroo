"""Test suite for the Paxml module."""

import jax.numpy as jnp


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return logits
