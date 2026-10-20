"""Test suite for the Flax Nnx module."""

import typing

import jax
import jax.numpy as jnp


def rnn_loop(
  cell: typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
  x: jnp.ndarray,
  init_state: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Helper to rnn loop.

  Args:
      cell: Test argument or fixture.
      x: Test argument or fixture.
      init_state: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  def scan_fn(carry: jnp.ndarray, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scans fn.

    Args:
        carry: Test argument or fixture.
        inputs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    out, next_state = cell(inputs, carry)
    return next_state, out

  final_state, outputs = jax.lax.scan(scan_fn, init_state, x)
  return outputs, final_state
