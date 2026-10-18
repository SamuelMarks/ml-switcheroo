"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


def rnn_loop(cell: typing.Any, x: tf.Tensor, init_state: tf.Tensor) -> typing.Any:
  """Helper to rnn loop.

  Args:
      cell: Test argument or fixture.
      x: Test argument or fixture.
      init_state: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  def scan_fn(carry: typing.Any, inputs: typing.Any) -> typing.Any:
    """Scans fn.

    Args:
        carry: Test argument or fixture.
        inputs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    out, next_state = cell(inputs, carry)
    return next_state

  pass
