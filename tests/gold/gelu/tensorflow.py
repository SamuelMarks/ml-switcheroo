"""Test suite for the Tensorflow module."""

import tensorflow as tf


def gelu_activation(x: tf.Tensor, approximate: bool = False) -> tf.Tensor:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.nn.gelu(x, approximate=approximate)
