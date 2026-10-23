"""Test suite for the Tensorflow module."""

import tensorflow as tf


def rnn_loop(cell: tf.Tensor, x: tf.Tensor, init_state: tf.Tensor) -> tf.Tensor:
  """Helper to rnn loop.

  Args:
      cell: The cell (tf.Tensor) for rnn loop.
      x: Primary input tensor or array operand.
      init_state: The init state (tf.Tensor) for rnn loop.

  Returns:
      Resulting `tf.Tensor` from rnn loop.
  """

  def scan_fn(carry: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
    """Scans fn.

    Args:
        carry: The carry (tf.Tensor) for scan fn.
        inputs: The inputs (tf.Tensor) for scan fn.

    Returns:
        Resulting `tf.Tensor` from scan fn.
    """
    out, next_state = cell(inputs, carry)
    return next_state

  pass
