"""Test suite for the Flax Nnx module."""

import typing

import jax


def rnn_loop(cell: typing.Any, x: typing.Any, init_state: typing.Any) -> tuple[typing.Any, typing.Any]:
  """Helper to rnn loop.

  Args:
      cell: Test argument or fixture.
      x: Test argument or fixture.
      init_state: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  def scan_fn(carry: typing.Any, inputs: typing.Any) -> tuple[typing.Any, typing.Any]:
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
