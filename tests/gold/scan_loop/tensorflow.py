"""Module docstring."""

import tensorflow as tf


def rnn_loop(cell, x: tf.Tensor, init_state: tf.Tensor):
  """Function docstring."""

  # <SWITCHEROO_FAILED_TO_TRANS>
  # TensorFlow uses tf.scan or tf.while_loop
  def scan_fn(carry, inputs):
    """Function docstring."""
    out, next_state = cell(inputs, carry)
    return next_state

  # Simplified representation
  # </SWITCHEROO_FAILED_TO_TRANS>
  pass
