"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def rnn_loop(cell, x: jnp.ndarray, init_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Helper to rnn loop."""

  def scan_fn(carry, inputs):
    """Scans fn."""
    (out, next_state) = cell(inputs, carry)
    return (next_state, out)

  (final_state, outputs) = jax.lax.scan(scan_fn, init_state, x)
  return (outputs, final_state)
