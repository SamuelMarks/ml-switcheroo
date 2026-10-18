"""Test suite for the Numpy module."""

import typing

import numpy as np


def train_step(
  model: typing.Any,
  optimizer: typing.Any,
  x: np.ndarray[typing.Any, typing.Any],
  y: np.ndarray[typing.Any, typing.Any],
  loss_fn: typing.Any,
) -> typing.Any:
  """Trains step.

  Args:
      model: Test argument or fixture.
      optimizer: Test argument or fixture.
      x: Test argument or fixture.
      y: Test argument or fixture.
      loss_fn: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  predictions = model(x)
  loss = loss_fn(predictions, y)
  return loss
