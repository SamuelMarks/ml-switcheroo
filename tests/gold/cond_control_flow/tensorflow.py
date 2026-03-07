"""Module docstring."""

import tensorflow as tf


def conditional_op(pred: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # TF uses tf.cond
  return tf.cond(pred, lambda: x * 2, lambda: x + 2)
  # </SWITCHEROO_FAILED_TO_TRANS>
