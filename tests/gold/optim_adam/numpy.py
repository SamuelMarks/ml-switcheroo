"""Test suite for the Numpy module."""

import typing


class NumpyAdamProtocol(typing.Protocol):
  """Protocol for NumpyAdam."""

  lr: float


def setup_adam(model: str | None = None, lr: float = 0.001) -> NumpyAdamProtocol:
  """Execute `NumpyAdam` to support setup adam in numpy tests.

  Args:
      model: Neural network model instance under test.
      lr: Input `lr` parameter value (`float`).

  Returns:
      Resulting `NumpyAdamProtocol` from setup adam.
  """

  class NumpyAdam:
    """Sentinel `NumpyAdam` fixture class used in numpy tests."""

    def __init__(self, lr: float) -> None:
      """Initializes the NumpyAdam instance.

      Args:
          lr: Input `lr` parameter value (`float`).
      """
      self.lr = lr

  return NumpyAdam(lr)
