"""Test suite for the Numpy module."""

import typing


def setup_adam(model: typing.Any, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam."""

  class NumpyAdam:
    """Test suite for the Numpy Adam component."""

    def __init__(self, lr: float) -> None:
      """Initializes the NumpyAdam instance."""
      self.lr = lr

  return NumpyAdam(lr)
