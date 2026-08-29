"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


def rnn_loop(cell: typing.Any, x: tf.Tensor, init_state: tf.Tensor) -> typing.Any:
  """Helper to rnn loop."""

  def scan_fn(carry: typing.Any, inputs: typing.Any) -> typing.Any:
    """Scans fn."""
    out, next_state = cell(inputs, carry)
    return next_state

  pass
