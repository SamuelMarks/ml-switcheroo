"""Test suite for the Flax Nnx module."""

import typing
import jax


def rnn_loop(cell: typing.Any, x: typing.Any, init_state: typing.Any) -> tuple[typing.Any, typing.Any]:
  """Helper to rnn loop."""

  def scan_fn(carry: typing.Any, inputs: typing.Any) -> tuple[typing.Any, typing.Any]:
    """Scans fn."""
    out, next_state = cell(inputs, carry)
    return next_state, out

  final_state, outputs = jax.lax.scan(scan_fn, init_state, x)
  return outputs, final_state
