"""Module docstring."""

import jax
import jax.numpy as jnp


def rnn_loop(cell, x: jnp.ndarray, init_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Function docstring."""

  # <SWITCHEROO_FAILED_TO_TRANS>
  # JAX uses lax.scan for compiled loops
  def scan_fn(carry, inputs):
    """Function docstring."""
    out, next_state = cell(inputs, carry)
    return next_state, out

  final_state, outputs = jax.lax.scan(scan_fn, init_state, x)
  return outputs, final_state
  # </SWITCHEROO_FAILED_TO_TRANS>
