"""Test suite for the Numpy module."""

import typing

import numpy as np
import numpy.typing


def train_step(
  model: typing.Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]],
  optimizer: typing.Optional[str],
  x: np.typing.NDArray[np.float64],
  y: np.typing.NDArray[np.float64],
  loss_fn: typing.Callable[
    [np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
    np.typing.NDArray[np.float64],
  ],
) -> np.typing.NDArray[np.float64]:
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
