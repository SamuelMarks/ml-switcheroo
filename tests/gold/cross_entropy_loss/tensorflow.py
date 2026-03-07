"""Module docstring."""

import tensorflow as tf


def compute_loss(logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
  # </SWITCHEROO_FAILED_TO_TRANS>
