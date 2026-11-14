"""Test suite for the Tensorflow module."""

import tensorflow as tf


def compute_loss(logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
  """Computes loss.

  Args:
      logits: Input `logits` parameter value (`tf.Tensor`).
      targets: Input `targets` parameter value (`tf.Tensor`).

  Returns:
      Resulting `tf.Tensor` from compute loss.
  """
  criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
