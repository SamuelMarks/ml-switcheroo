"""Keras 3 golden reference for global-norm gradient clipping."""

import keras


def clip_grads(
  grads: list[keras.KerasTensor],
  max_norm: float = 1.0,
) -> list[keras.KerasTensor]:
  """Clip a list of gradient tensors by global L2 norm in Keras 3.

  Args:
      grads: Sequence of gradient tensors to clip.
      max_norm: Maximum global L2 norm threshold.

  Returns:
      Clipped list of gradient tensors.
  """
  global_norm = keras.ops.sqrt(sum(keras.ops.sum(keras.ops.square(g)) for g in grads))
  scale = max_norm / keras.ops.maximum(global_norm, max_norm)
  return [g * scale for g in grads]
