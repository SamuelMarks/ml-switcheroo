"""Test suite for the Keras3 module."""

import keras


def compute_loss(logits: keras.KerasTensor, targets: keras.KerasTensor) -> keras.KerasTensor:
  """Computes loss.

  Args:
      logits: Input `logits` parameter value (`keras.KerasTensor`).
      targets: Input `targets` parameter value (`keras.KerasTensor`).

  Returns:
      Resulting `keras.KerasTensor` from compute loss.
  """
  criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
