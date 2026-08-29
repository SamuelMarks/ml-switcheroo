"""Test suite for the Keras3 module."""

import typing

import keras


def compute_loss(logits: typing.Any, targets: typing.Any) -> typing.Any:
  """Computes loss."""
  criterion: typing.Any = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # type: ignore
  return criterion(targets, logits)
