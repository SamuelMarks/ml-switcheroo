"""Test suite for the Tensorflow module."""

import tensorflow as tf


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> tf.Tensor:
  """Gets clipped optimizer.

  Args:
      learning_rate: Input `learning_rate` parameter value (`float`).
      max_norm: Input `max_norm` parameter value (`float`).

  Returns:
      Resulting `tf.Tensor` from get clipped optimizer.
  """
  return tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=max_norm)


def clip_grads(grads: list[tf.Tensor], max_norm: float = 1.0) -> list[tf.Tensor]:
  """Clip a list of TensorFlow gradient tensors by global norm via ``tf.clip_by_global_norm``.

  Args:
      grads: List of ``tf.Tensor`` gradients.
      max_norm: Maximum global L2 norm threshold.

  Returns:
      Clipped list of ``tf.Tensor`` gradients.
  """
  clipped, _ = tf.clip_by_global_norm(grads, max_norm)
  res: list[tf.Tensor] = list(clipped)
  return res
