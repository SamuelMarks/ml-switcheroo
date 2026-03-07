"""Module docstring."""

import keras


def compute_loss(logits, targets):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # SparseCategoricalCrossentropy handles integer targets
  criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
  # </SWITCHEROO_FAILED_TO_TRANS>
