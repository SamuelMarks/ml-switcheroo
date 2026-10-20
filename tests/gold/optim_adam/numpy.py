"""Test suite for the Numpy module."""

import typing


class NumpyAdamProtocol(typing.Protocol):
  """Protocol for NumpyAdam."""

  lr: float


def setup_adam(model: typing.Optional[str] = None, lr: float = 0.001) -> NumpyAdamProtocol:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  class NumpyAdam:
    """Test helper class NumpyAdam."""

    def __init__(self, lr: float) -> None:
      """Initializes the NumpyAdam instance.

      Args:
          lr: Test argument or fixture.
      """
      self.lr = lr

  return NumpyAdam(lr)
