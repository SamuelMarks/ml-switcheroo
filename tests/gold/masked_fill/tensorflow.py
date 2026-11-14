"""TensorFlow golden reference for attention score masked fill."""

import tensorflow as tf


def causal_mask_fill(
  scores: tf.Tensor,
  mask: tf.Tensor,
  value: float = -1000000000.0,
) -> tf.Tensor:
  """Replace positions where ``mask == 0`` with ``value`` using ``tf.where``.

  Args:
      scores: Unmasked ``tf.Tensor`` attention scores.
      mask: Mask tensor where ``0`` denotes masked positions.
      value: Scalar fill value for masked entries.

  Returns:
      Masked ``tf.Tensor`` attention scores.
  """
  fill = tf.cast(value, scores.dtype)
  res: tf.Tensor = tf.where(tf.equal(mask, 0), fill, scores)
  return res
