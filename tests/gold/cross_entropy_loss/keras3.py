"""Test suite for the Keras3 module."""

import keras


def compute_loss(logits: keras.KerasTensor, targets: keras.KerasTensor) -> keras.KerasTensor:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
