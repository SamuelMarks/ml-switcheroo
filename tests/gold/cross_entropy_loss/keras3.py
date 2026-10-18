"""Test suite for the Keras3 module."""

import typing

import keras


def compute_loss(logits: typing.Any, targets: typing.Any) -> typing.Any:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  criterion: typing.Any = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  return criterion(targets, logits)
