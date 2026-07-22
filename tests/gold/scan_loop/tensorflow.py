"""Test suite for the Tensorflow module."""

import tensorflow as tf


def rnn_loop(cell, x: tf.Tensor, init_state: tf.Tensor):
  """Helper to rnn loop."""

  def scan_fn(carry, inputs):
    """Scans fn."""
    (out, next_state) = cell(inputs, carry)
    return next_state

  pass
