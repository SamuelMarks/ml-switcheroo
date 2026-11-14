"""Test suite for the Flax Nnx module."""

import typing

import jax
import jax.numpy as jnp


def rnn_loop(
  cell: typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
  x: jnp.ndarray,
  init_state: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Execute `scan` to support rnn loop in flax NNX tests.

  Args:
      cell: Input `cell` parameter value (`typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]`).
      x: Primary input tensor or array operand.
      init_state: Input `init_state` parameter value (`jnp.ndarray`).

  Returns:
      Resulting `tuple[jnp.ndarray, jnp.ndarray]` from rnn loop.
  """

  def scan_fn(carry: jnp.ndarray, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scans fn.

    Args:
        carry: Input `carry` parameter value (`jnp.ndarray`).
        inputs: Input `inputs` parameter value (`jnp.ndarray`).

    Returns:
        Resulting `tuple[jnp.ndarray, jnp.ndarray]` from scan fn.
    """
    out, next_state = cell(inputs, carry)
    return next_state, out

  final_state, outputs = jax.lax.scan(scan_fn, init_state, x)
  return outputs, final_state
