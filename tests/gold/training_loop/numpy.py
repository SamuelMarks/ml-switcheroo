"""Test suite for the Numpy module."""

import typing

import numpy as np


def train_step(model: typing.Any, optimizer: typing.Any, x: np.ndarray, y: np.ndarray, loss_fn: typing.Any) -> typing.Any:
  """Trains step."""
  predictions = model(x)
  loss = loss_fn(predictions, y)
  return loss
