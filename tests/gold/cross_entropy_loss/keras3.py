"""Test suite for the Keras3 module."""

import keras


def compute_loss(logits, targets):
  """Computes loss."""
  criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
