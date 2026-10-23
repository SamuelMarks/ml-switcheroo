"""Test suite for the Keras3 module."""

import keras


def compute_loss(logits: keras.KerasTensor, targets: keras.KerasTensor) -> keras.KerasTensor:
  """Computes loss.

  Args:
      logits: The logits (keras.KerasTensor) for compute loss.
      targets: The targets (keras.KerasTensor) for compute loss.

  Returns:
      Resulting `keras.KerasTensor` from compute loss.
  """
  criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
