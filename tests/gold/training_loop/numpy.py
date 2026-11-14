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
      optimizer: Input `optimizer` parameter value (`str | None`).
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: Input `loss_fn` parameter value (`typing.Callable[[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]`).

  Returns:
      Resulting `np.typing.NDArray[np.float64]` from train step.
  """
  predictions = model(x)
  return loss_fn(predictions, y)
