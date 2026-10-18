"""Test suite for the Numpy module."""

import typing


def setup_adam(model: typing.Any, lr: float = 0.001) -> typing.Any:
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
