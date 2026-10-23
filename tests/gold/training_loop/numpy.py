"""Test suite for the Numpy module."""

import typing

import numpy as np
import numpy.typing


def train_step(
  model: typing.Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]],
  optimizer: str | None,
  x: np.typing.NDArray[np.float64],
  y: np.typing.NDArray[np.float64],
  loss_fn: typing.Callable[
    [np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
    np.typing.NDArray[np.float64],
  ],
) -> np.typing.NDArray[np.float64]:
  """Trains step.

  Args:
      model: Neural network model instance under test.
      optimizer: The optimizer (str | None) for train step.
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: The loss fn (typing.Callable[[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]) for train step.

  Returns:
      Resulting `np.typing.NDArray[np.float64]` from train step.
  """
  predictions = model(x)
  loss = loss_fn(predictions, y)
  return loss
