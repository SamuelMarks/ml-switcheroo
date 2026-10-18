"""Test suite for the Tensorflow module."""

import tensorflow as tf


def compute_loss(logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
