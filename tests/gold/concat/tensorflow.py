"""Test suite for the Tensorflow module."""

import tensorflow as tf


def concat_tensors(x: tf.Tensor, y: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.concat([x, y], axis=axis)
