"""Test suite for the Tensorflow module."""

import tensorflow as tf


def rnn_loop(cell: tf.Tensor, x: tf.Tensor, init_state: tf.Tensor) -> tf.Tensor:
  """Execute `cell` to support rnn loop in tensorflow tests.

  Args:
      cell: Input `cell` parameter value (`tf.Tensor`).
      x: Primary input tensor or array operand.
      init_state: Input `init_state` parameter value (`tf.Tensor`).

  Returns:
      Resulting `tf.Tensor` from rnn loop.
  """

  def scan_fn(carry: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
    """Scans fn.

    Args:
        carry: Input `carry` parameter value (`tf.Tensor`).
        inputs: Input `inputs` parameter value (`tf.Tensor`).

    Returns:
        Resulting `tf.Tensor` from scan fn.
    """
    _out, next_state = cell(inputs, carry)
    return next_state
