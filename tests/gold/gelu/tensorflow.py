"""Module docstring."""

import tensorflow as tf


def gelu_activation(x: tf.Tensor, approximate: bool = False) -> tf.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return tf.nn.gelu(x, approximate=approximate)
  # </SWITCHEROO_FAILED_TO_TRANS>
