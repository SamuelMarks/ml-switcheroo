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
      cell: The cell (typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]) for rnn loop.
      x: Primary input tensor or array operand.
      init_state: The init state (jnp.ndarray) for rnn loop.

  Returns:
      Resulting `tuple[jnp.ndarray, jnp.ndarray]` from rnn loop.
  """

  def scan_fn(carry: jnp.ndarray, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scans fn.

    Args:
        carry: The carry (jnp.ndarray) for scan fn.
        inputs: The inputs (jnp.ndarray) for scan fn.

    Returns:
        Resulting `tuple[jnp.ndarray, jnp.ndarray]` from scan fn.
    """
    out, next_state = cell(inputs, carry)
    return next_state, out

  final_state, outputs = jax.lax.scan(scan_fn, init_state, x)
  return outputs, final_state
