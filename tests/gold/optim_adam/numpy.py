"""Test suite for the Numpy module."""

import typing


class NumpyAdamProtocol(typing.Protocol):
  """Protocol for NumpyAdam."""

  lr: float


def setup_adam(model: str | None = None, lr: float = 0.001) -> NumpyAdamProtocol:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `NumpyAdamProtocol` from setup adam.
  """

  class NumpyAdam:
    """Test helper class NumpyAdam."""

    def __init__(self, lr: float) -> None:
      """Initializes the NumpyAdam instance.

      Args:
          lr: The lr (float) for   init  .
      """
      self.lr = lr

  return NumpyAdam(lr)
