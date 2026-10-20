"""Test suite for the Tensorflow module."""

import tensorflow as tf


def rnn_loop(cell: tf.Tensor, x: tf.Tensor, init_state: tf.Tensor) -> tf.Tensor:
  """Helper to rnn loop.

  Args:
      cell: Test argument or fixture.
      x: Test argument or fixture.
      init_state: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  def scan_fn(carry: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
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
